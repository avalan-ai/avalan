#!/usr/bin/env python
"""Validate the frozen, dormant patch acceptance contract."""

from argparse import ArgumentParser, Namespace
from ast import (
    AST,
    Add,
    AnnAssign,
    Assign,
    AsyncFunctionDef,
    Attribute,
    BinOp,
    Call,
    ClassDef,
    Constant,
    Div,
    FormattedValue,
    FunctionDef,
    Import,
    ImportFrom,
    JoinedStr,
    List,
    Name,
    Tuple,
    alias,
    expr,
    get_source_segment,
    walk,
)
from ast import parse as parse_python
from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from logging import getLogger
from os import environ
from pathlib import Path, PurePosixPath
from re import compile as compile_regex
from sys import addaudithook, stderr
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
from verify_patch_types import PatchTypeContractError
from verify_patch_types import load_manifest as load_type_manifest
from verify_src_coverage import CoverageVerificationError, verify_src_coverage

_FEATURE = "patch"
_CURRENT_PHASE = 0
_MAX_PHASE = 15
_PINNED_ACCEPTANCE_HISTORY_SNAPSHOT_SHA256 = (
    "6ce51772dc60d33aec7f69129edd45ab0792ef5eea788071a3b5ad577081dcd6"
)
_FIXTURE_NAMES = (
    "requirements_traceability.json",
    "contract_decisions.json",
    "acceptance_manifest.json",
    "acceptance_history.json",
    "failure_matrix.json",
    "target_conformance.json",
    "surface_conformance.json",
    "source_symbols.json",
    "type_contract_manifest.json",
    "goldens.json",
    "threat_model.json",
    "baseline_evidence.json",
    "phase_evidence.json",
)
_PHASE_EVIDENCE_ARTIFACT_ENVS = (
    "AVALAN_PATCH_PHASE_EVIDENCE_COVERAGE_JSON",
    "AVALAN_PATCH_PHASE_EVIDENCE_COVERAGE_XML",
    "AVALAN_PATCH_PHASE_EVIDENCE_PYTEST_FACTS",
)
_NODE_PATTERN = compile_regex(r"^tests/[A-Za-z0-9_./-]+\.py::[^\s]+$")
_IDENTIFIER_PATTERN = compile_regex(r"^PATCH-[A-Z0-9-]+$")
_SYMBOL_PATTERN = compile_regex(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256_PATTERN = compile_regex(r"^[0-9a-f]{64}$")
_REQUIREMENT_MODALITIES = frozenset(
    (
        "MUST",
        "MUST_NOT",
        "SHOULD",
        "MAY",
        "NON_GOAL",
        "DECISION",
        "ACCEPTANCE",
        "DOCUMENTATION",
        "APPENDIX",
        "CONTRACT",
    )
)
_REQUIREMENT_RECORD_LAYOUT = (
    "id",
    "source_section",
    "source_start_line",
    "source_end_line",
    "normalized_statement",
    "modality",
    "owning_phase",
    "optional_feature_disposition",
    "source_kind",
    "implementation_artifact",
    "implementation_symbol",
    "test_node_id",
    "documentation_owner",
    "evidence_class",
)
_MAX_ACTIVE_REQUIREMENTS_PER_NODE = 8
_ACTIVE_IMPLEMENTATION_ROOT = "scripts/"
_PLANNED_IMPLEMENTATION_ROOTS = ("src/", "docs/")
_REQUIREMENT_SOURCE_KINDS = frozenset(
    (
        "acceptance",
        "appendix",
        "contract",
        "documentation",
        "locked_decision",
        "non_goal",
        "normative",
    )
)
_SOURCE_SECTION_PATTERN = compile_regex(
    r"^(?:[3-9]|[12][0-9]|3[01])(?:\.[1-9][0-9]?)?$|^A\.[1-4]$"
)
_REQUIRED_SOURCE_SECTIONS = frozenset(str(value) for value in range(3, 32))
_REQUIRED_APPENDIX_SECTIONS = frozenset(("A.1", "A.2", "A.3", "A.4"))
_INVENTORY_AREAS = frozenset(
    (
        "src",
        "coverage_exclusions",
        "public_tools",
        "events",
        "migrations",
        "server_routes",
        "protocols",
        "context_capabilities",
    )
)
_NODE_CATEGORIES = frozenset(
    (
        "positive",
        "negative",
        "race",
        "cancellation",
        "crash",
        "security",
        "privacy",
        "performance",
        "type",
        "integration",
        "context_conformance",
        "public_e2e",
    )
)
_SURFACE_IDS = frozenset(
    (
        "sdk",
        "cli",
        "server",
        "mcp",
        "a2a",
        "flow",
        "task",
        "multi_agent",
        "json_function",
        "freeform",
    )
)
_TARGET_CONTEXTS = frozenset(("local", "sandbox", "container"))
_FORBIDDEN_SOURCE_ARTIFACTS = frozenset(
    (
        "specs/PATCH.md",
        "specs/PATCH-agenda.md",
    )
)
_SOURCE_ARTIFACT_AUDIT_ENABLED: ContextVar[bool] = ContextVar(
    "patch_source_artifact_audit_enabled",
    default=False,
)
_LIFECYCLE_STATES = (
    "received",
    "parsed",
    "scope_bound",
    "preflight_authorized",
    "planned",
    "approval_required",
    "approved",
    "commit_ready",
    "commit_started",
    "settlement_pending",
    "settled",
    "request_completed",
)
_LIFECYCLE_TRANSITIONS = (
    ("received", "parsed", "complete bounded input"),
    (
        "received",
        "request_completed",
        "parse, lexical, cancellation, or limit failure",
    ),
    (
        "parsed",
        "request_completed",
        "parse, lexical, cancellation, or limit failure",
    ),
    ("parsed", "scope_bound", "trusted scope bound"),
    (
        "scope_bound",
        "preflight_authorized",
        (
            "pre-inspection policy allows the conservative external-effect "
            "upper bound, paths, precondition observation, and read budget"
        ),
    ),
    (
        "scope_bound",
        "request_completed",
        "pre-inspection denial or unavailable capability",
    ),
    (
        "preflight_authorized",
        "planned",
        "bounded snapshot and virtual planning succeed",
    ),
    (
        "preflight_authorized",
        "request_completed",
        "snapshot, match, conflict, content, or limit failure",
    ),
    ("planned", "request_completed", "exact final-effect policy denies"),
    ("planned", "approval_required", "policy requires review"),
    ("planned", "approved", "trusted class is preauthorized"),
    ("approval_required", "approved", "grant received and bound"),
    (
        "approval_required",
        "request_completed",
        "denial, expiry, cancellation, or unavailable broker",
    ),
    ("approved", "commit_ready", "lock and final revalidation succeed"),
    (
        "approved",
        "request_completed",
        "stale, cancellation, timeout, or policy change",
    ),
    (
        "commit_ready",
        "commit_started",
        (
            "first requested effect or context-visible staging artifact may "
            "become visible"
        ),
    ),
    ("commit_started", "settled", "worker settles and journal is reconciled"),
    (
        "commit_started",
        "settlement_pending",
        "worker is still live or not yet provably fenced",
    ),
    (
        "settlement_pending",
        "settled",
        "same worker settles or is fenced and reconciled",
    ),
    ("settled", "request_completed", "result projection complete"),
)


def _limit_composition(identifier: str, observation: str) -> str:
    """Return the required narrow effective-limit composition expression."""
    return (
        f"effective.{identifier} = min(provider.{identifier}, "
        f"manager.{identifier}, policy.{identifier}, context.{identifier}, "
        f"backend.{identifier}); {observation} <= effective.{identifier}"
    )


_LIMIT_COMPOSITION = {
    "max_raw_input_bytes": _limit_composition(
        "max_raw_input_bytes", "raw_input_bytes"
    ),
    "max_raw_input_lines": _limit_composition(
        "max_raw_input_lines", "raw_input_lines"
    ),
    "max_path_count": _limit_composition("max_path_count", "path_count"),
    "max_path_length": _limit_composition("max_path_length", "path_length"),
    "max_path_component_length": _limit_composition(
        "max_path_component_length", "path_component_length"
    ),
    "max_path_depth": _limit_composition("max_path_depth", "path_depth"),
    "max_file_count": _limit_composition("max_file_count", "file_count"),
    "max_file_lineages": _limit_composition(
        "max_file_lineages", "file_lineage_count"
    ),
    "max_operations": _limit_composition("max_operations", "operation_count"),
    "max_update_declarations": _limit_composition(
        "max_update_declarations", "update_declaration_count"
    ),
    "max_hunks": _limit_composition("max_hunks", "hunk_count"),
    "max_replacements": _limit_composition(
        "max_replacements", "replacement_count"
    ),
    "max_match_candidates": _limit_composition(
        "max_match_candidates", "match_candidate_count"
    ),
    "max_per_file_snapshot_bytes": _limit_composition(
        "max_per_file_snapshot_bytes", "per_file_snapshot_bytes"
    ),
    "max_aggregate_bytes_read": _limit_composition(
        "max_aggregate_bytes_read", "aggregate_bytes_read"
    ),
    "max_per_file_proposed_bytes": _limit_composition(
        "max_per_file_proposed_bytes", "per_file_proposed_bytes"
    ),
    "max_aggregate_proposed_state_bytes": _limit_composition(
        "max_aggregate_proposed_state_bytes", "aggregate_proposed_state_bytes"
    ),
    "max_added_bytes": _limit_composition("max_added_bytes", "added_bytes"),
    "max_removed_bytes": _limit_composition(
        "max_removed_bytes", "removed_bytes"
    ),
    "max_total_changed_bytes": _limit_composition(
        "max_total_changed_bytes", "total_changed_bytes"
    ),
    "max_diff_work_units": _limit_composition(
        "max_diff_work_units", "diff_work_units"
    ),
    "max_full_review_diff_bytes": _limit_composition(
        "max_full_review_diff_bytes", "full_review_diff_bytes"
    ),
    "max_returned_diff_bytes": _limit_composition(
        "max_returned_diff_bytes", "returned_diff_bytes"
    ),
    "max_planning_memory_bytes": _limit_composition(
        "max_planning_memory_bytes", "planning_memory_bytes"
    ),
    "max_private_staging_disk_bytes": _limit_composition(
        "max_private_staging_disk_bytes", "private_staging_disk_bytes"
    ),
    "max_journal_bytes": _limit_composition(
        "max_journal_bytes", "journal_bytes"
    ),
    "max_planning_duration_ticks": _limit_composition(
        "max_planning_duration_ticks", "planning_duration_ticks"
    ),
    "max_approval_wait_ticks": _limit_composition(
        "max_approval_wait_ticks", "approval_wait_ticks"
    ),
    "max_lock_wait_ticks": _limit_composition(
        "max_lock_wait_ticks", "lock_wait_ticks"
    ),
    "max_revalidation_duration_ticks": _limit_composition(
        "max_revalidation_duration_ticks", "revalidation_duration_ticks"
    ),
    "max_commit_duration_ticks": _limit_composition(
        "max_commit_duration_ticks", "commit_duration_ticks"
    ),
    "max_verification_duration_ticks": _limit_composition(
        "max_verification_duration_ticks", "verification_duration_ticks"
    ),
    "max_diagnostic_duration_ticks": _limit_composition(
        "max_diagnostic_duration_ticks", "diagnostic_duration_ticks"
    ),
    "max_concurrent_plans": _limit_composition(
        "max_concurrent_plans", "concurrent_plan_count"
    ),
    "max_denied_attempts": _limit_composition(
        "max_denied_attempts", "denied_attempt_count"
    ),
    "max_failed_attempts": _limit_composition(
        "max_failed_attempts", "failed_attempt_count"
    ),
    "max_retransmission_attempts": _limit_composition(
        "max_retransmission_attempts", "retransmission_attempt_count"
    ),
}
_RESOURCE_DEPTH_FIELDS = (
    "transaction",
    "coordinator_lease",
    "target_handle",
    "target_worker",
    "staging_resource",
    "approval_wait",
)
_AWAIT_MATRIX = {
    "store_connection": (1, 0, 0, 0, 0, 0),
    "fault_wait": (0, 0, 0, 0, 0, 0),
    "target_negotiation": (0, 0, 0, 0, 0, 0),
    "target_inspection": (0, 0, 0, 0, 0, 0),
    "target_precondition": (0, 0, 0, 0, 0, 0),
    "target_handle_open": (0, 0, 0, 0, 0, 0),
    "target_handle_close": (0, 0, 1, 0, 0, 0),
    "target_lock_acquire": (0, 0, 1, 0, 0, 0),
    "target_lock_release": (0, 1, 1, 0, 0, 0),
    "target_stage": (0, 1, 1, 0, 0, 0),
    "target_cleanup": (0, 1, 1, 0, 1, 0),
    "target_namespace_mutation": (0, 0, 0, 0, 0, 0),
    "target_commit": (0, 1, 1, 1, 1, 0),
    "target_verification": (0, 1, 1, 0, 1, 0),
    "approval_decision": (0, 0, 0, 0, 0, 1),
    "approval_consume": (0, 0, 0, 0, 0, 1),
    "approval_concurrent_consume": (0, 0, 0, 0, 0, 1),
    "target_factory_create": (0, 0, 0, 0, 0, 0),
    "target_factory_negotiate": (0, 0, 0, 0, 0, 0),
    "publication": (0, 0, 0, 0, 0, 0),
}
_REPLACEMENT_REQUIREMENTS = (
    "versioned_replacement_proposal",
    "threat_review",
    "migration_plan",
    "traceability_update",
    "full_acceptance_evidence",
)
_FAILURE_BOUNDARY_CATALOG = (
    "lifecycle.received",
    "lifecycle.scope_bound",
    "lifecycle.preinspection_authorized",
    "lifecycle.planned",
    "lifecycle.awaiting_approval",
    "lifecycle.commit_owner_assigned",
    "lifecycle.commit_started",
    "lifecycle.reconciling",
    "lifecycle.request_completed",
    "target.negotiate_capabilities",
    "target.inspect",
    "target.observe_precondition",
    "target.open_handle",
    "target.close_handle",
    "target.acquire_lock",
    "target.release_lock",
    "target.stage_artifact",
    "target.namespace_mutation",
    "target.commit_step",
    "target.verify",
    "requested_effect.step_before",
    "requested_effect.step_after",
    "store.reserve_request",
    "store.persist_plan",
    "store.consume_grant",
    "store.assign_commit_owner",
    "store.record_commit_started",
    "store.journal_step",
    "store.publish_terminal",
    "approval.decide",
    "approval.consume",
    "approval.concurrent_consume",
    "commit.intent_fence",
    "commit.target_step",
    "artifact.stage",
    "artifact.verify",
    "cleanup.staging",
    "cancellation.before_commit",
    "cancellation.after_commit",
    "timeout.before_commit",
    "timeout.after_commit",
    "disconnect.before_commit",
    "disconnect.after_commit",
    "publication.outbox",
    "publication.result",
    "publication.event",
)
_FAILURE_STEP_STATES = frozenset(
    (
        "not_committed",
        "committed",
        "indeterminate",
    )
)
_FAILURE_LINEAGE_STATES = frozenset(
    (
        "not_committed",
        "committed",
        "partially_committed",
        "indeterminate",
    )
)
_FAILURE_ARTIFACT_STATES = frozenset(
    (
        "absent",
        "staged",
        "cleaned",
        "leaked",
        "unknown",
    )
)
_FAILURE_EFFECT_FACTS = frozenset(
    (
        "not_committed",
        "committed",
        "partially_committed",
        "indeterminate",
    )
)
_FAILURE_WORKSPACE_FACTS = frozenset(("unchanged", "changed", "unknown"))
_FAILURE_PENDING_BEHAVIORS = frozenset(
    (
        "terminal_not_pending",
        "pending_reconciliation",
        "pending_caller_detached",
    )
)
_FAILURE_RETRYABILITIES = frozenset(
    (
        "retryable_precommit",
        "retransmit_only",
        "not_retryable",
    )
)
_FAILURE_PROJECTIONS = frozenset(
    (
        "patch_approval_denied",
        "patch_cancelled",
        "patch_commit_failed",
        "patch_committed",
        "patch_contract_invalid",
        "patch_incapable",
        "patch_indeterminate",
        "patch_pending",
        "patch_rejected",
        "patch_timeout",
    )
)
_FAILURE_EVENTS = frozenset(
    (
        "request_received",
        "scope_bound",
        "preinspection_authorized",
        "plan_frozen",
        "review_requested",
        "commit_owner_assigned",
        "commit_started",
        "reconciling",
        "request_completed",
    )
)


@dataclass(frozen=True, kw_only=True, slots=True)
class _FailureSemantic:
    """Define the complete expected truth for one frozen failure boundary."""

    commit_started: bool
    counts: tuple[int, int, int, int, int]
    step_state: str
    lineage_state: str
    artifact_state: str
    requested_effect_fact: str
    workspace_change_fact: str
    workspace_oracle_equal: bool
    events: tuple[str, ...]
    pending_behavior: str
    retryability: str
    public_projection: str


_RECEIVED = ("request_received",)
_SCOPED = (*_RECEIVED, "scope_bound")
_AUTHORIZED = (*_SCOPED, "preinspection_authorized")
_PLANNED = (*_AUTHORIZED, "plan_frozen")
_COMMIT_OWNER = (*_PLANNED, "commit_owner_assigned")
_COMMIT_STARTED = (*_COMMIT_OWNER, "commit_started")
_RECONCILING = (*_COMMIT_STARTED, "reconciling")


def _terminal(events: tuple[str, ...]) -> tuple[str, ...]:
    """Append the sole terminal lifecycle event to a settled sequence."""
    return (*events, "request_completed")


def _precommit_semantic(
    *,
    events: tuple[str, ...],
    counts: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0),
    artifact_state: str = "absent",
    projection: str = "patch_rejected",
) -> _FailureSemantic:
    """Return the exact zero-write semantics before commit begins."""
    return _FailureSemantic(
        commit_started=False,
        counts=counts,
        step_state="not_committed",
        lineage_state="not_committed",
        artifact_state=artifact_state,
        requested_effect_fact="not_committed",
        workspace_change_fact="unchanged",
        workspace_oracle_equal=True,
        events=_terminal(events),
        pending_behavior="terminal_not_pending",
        retryability="retryable_precommit",
        public_projection=projection,
    )


def _commit_failed_semantic(
    *,
    counts: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0),
    artifact_state: str = "cleaned",
    workspace_change_fact: str = "unchanged",
) -> _FailureSemantic:
    """Return the exact settled truth after commit begins without an effect."""
    return _FailureSemantic(
        commit_started=True,
        counts=counts,
        step_state="not_committed",
        lineage_state="not_committed",
        artifact_state=artifact_state,
        requested_effect_fact="not_committed",
        workspace_change_fact=workspace_change_fact,
        workspace_oracle_equal=workspace_change_fact == "unchanged",
        events=_terminal(_RECONCILING),
        pending_behavior="terminal_not_pending",
        retryability="not_retryable",
        public_projection="patch_commit_failed",
    )


def _committed_semantic(
    *,
    counts: tuple[int, int, int, int, int],
    artifact_state: str = "cleaned",
) -> _FailureSemantic:
    """Return the exact settled truth after one requested effect commits."""
    return _FailureSemantic(
        commit_started=True,
        counts=counts,
        step_state="committed",
        lineage_state="committed",
        artifact_state=artifact_state,
        requested_effect_fact="committed",
        workspace_change_fact="changed",
        workspace_oracle_equal=False,
        events=_terminal(_RECONCILING),
        pending_behavior="terminal_not_pending",
        retryability="not_retryable",
        public_projection="patch_committed",
    )


def _pending_semantic(
    *,
    counts: tuple[int, int, int, int, int],
) -> _FailureSemantic:
    """Return the nonterminal truth while a postcommit worker settles."""
    return _FailureSemantic(
        commit_started=True,
        counts=counts,
        step_state="indeterminate",
        lineage_state="indeterminate",
        artifact_state="unknown",
        requested_effect_fact="indeterminate",
        workspace_change_fact="unknown",
        workspace_oracle_equal=False,
        events=_RECONCILING,
        pending_behavior="pending_reconciliation",
        retryability="retransmit_only",
        public_projection="patch_pending",
    )


def _failure_semantics_catalog() -> dict[str, _FailureSemantic]:
    """Return all 46 boundary-specific failure truths without inference."""
    pre = _precommit_semantic
    commit_failed = _commit_failed_semantic
    committed = _committed_semantic
    pending = _pending_semantic
    return {
        "lifecycle.received": pre(events=_RECEIVED),
        "lifecycle.scope_bound": pre(events=_SCOPED),
        "lifecycle.preinspection_authorized": pre(events=_AUTHORIZED),
        "lifecycle.planned": pre(events=_PLANNED),
        "lifecycle.awaiting_approval": pre(
            events=(*_PLANNED, "review_requested"),
            projection="patch_approval_denied",
        ),
        "lifecycle.commit_owner_assigned": pre(events=_COMMIT_OWNER),
        "lifecycle.commit_started": commit_failed(),
        "lifecycle.reconciling": pending(counts=(0, 0, 0, 0, 0)),
        "lifecycle.request_completed": pre(events=_RECEIVED),
        "target.negotiate_capabilities": pre(
            events=_SCOPED, projection="patch_incapable"
        ),
        "target.inspect": pre(events=_AUTHORIZED, counts=(1, 0, 0, 0, 0)),
        "target.observe_precondition": pre(
            events=_AUTHORIZED, counts=(1, 0, 0, 0, 0)
        ),
        "target.open_handle": pre(events=_COMMIT_OWNER),
        "target.close_handle": pre(events=_COMMIT_OWNER),
        "target.acquire_lock": pre(events=_COMMIT_OWNER),
        "target.release_lock": pre(events=_COMMIT_OWNER),
        "target.stage_artifact": pre(
            events=_COMMIT_OWNER, artifact_state="cleaned"
        ),
        "target.namespace_mutation": commit_failed(
            counts=(0, 1, 0, 0, 0),
            artifact_state="leaked",
            workspace_change_fact="changed",
        ),
        "target.commit_step": committed(counts=(0, 1, 0, 0, 1)),
        "target.verify": committed(counts=(0, 0, 0, 0, 1)),
        "requested_effect.step_before": pre(events=_COMMIT_OWNER),
        "requested_effect.step_after": committed(counts=(0, 1, 0, 0, 1)),
        "store.reserve_request": pre(events=_RECEIVED),
        "store.persist_plan": pre(events=_PLANNED),
        "store.consume_grant": pre(events=_PLANNED),
        "store.assign_commit_owner": pre(events=_COMMIT_OWNER),
        "store.record_commit_started": commit_failed(),
        "store.journal_step": pending(counts=(0, 0, 0, 0, 1)),
        "store.publish_terminal": committed(counts=(0, 0, 0, 0, 1)),
        "approval.decide": pre(
            events=(*_PLANNED, "review_requested"),
            counts=(0, 0, 0, 1, 0),
            projection="patch_approval_denied",
        ),
        "approval.consume": pre(
            events=(*_PLANNED, "review_requested"),
            counts=(0, 0, 0, 1, 0),
            projection="patch_approval_denied",
        ),
        "approval.concurrent_consume": pre(
            events=(*_PLANNED, "review_requested"),
            counts=(0, 0, 0, 1, 0),
            projection="patch_approval_denied",
        ),
        "commit.intent_fence": pre(events=_COMMIT_OWNER),
        "commit.target_step": committed(counts=(0, 1, 0, 0, 1)),
        "artifact.stage": pre(events=_COMMIT_OWNER, artifact_state="cleaned"),
        "artifact.verify": committed(
            counts=(0, 0, 0, 0, 1), artifact_state="unknown"
        ),
        "cleanup.staging": committed(
            counts=(0, 0, 0, 0, 1), artifact_state="leaked"
        ),
        "cancellation.before_commit": pre(
            events=_COMMIT_OWNER, projection="patch_cancelled"
        ),
        "cancellation.after_commit": pending(counts=(0, 1, 0, 0, 1)),
        "timeout.before_commit": pre(
            events=_COMMIT_OWNER, projection="patch_timeout"
        ),
        "timeout.after_commit": pending(counts=(0, 1, 0, 0, 1)),
        "disconnect.before_commit": pre(events=_COMMIT_OWNER),
        "disconnect.after_commit": pending(counts=(0, 1, 0, 0, 1)),
        "publication.outbox": committed(counts=(0, 0, 0, 0, 1)),
        "publication.result": committed(counts=(0, 0, 0, 0, 1)),
        "publication.event": committed(counts=(0, 0, 0, 0, 1)),
    }


_FAILURE_SEMANTICS = _failure_semantics_catalog()
_GOLDEN_CATEGORIES = frozenset(
    (
        "grammar",
        "path",
        "text",
        "matching",
        "lineage",
        "diff",
        "fingerprint",
        "result",
        "error",
        "event",
        "redaction",
        "wire",
    )
)
_THREAT_IDS = frozenset(
    (
        "malicious_workspace_content",
        "target_replacement",
        "protocol_replay",
        "authority_swap",
        "renderer_injection",
        "resource_exhaustion",
    )
)


class PatchAcceptanceError(RuntimeError):
    """Report a malformed or incomplete dormant patch acceptance contract."""


@dataclass(frozen=True, kw_only=True, slots=True)
class AcceptanceNode:
    """Store one lifecycle-aware exact pytest acceptance node."""

    identifier: str
    category: str
    lifecycle: str
    active_from_phase: int
    requirement_ids: tuple[str, ...]
    node_id: str
    surface: str
    context: str
    platform: str
    operation: str
    permission_label: str
    commit_boundary: str
    evidence_class: str


@dataclass(frozen=True, kw_only=True, slots=True)
class RequirementOwnership:
    """Store one requirement's implementation and executable evidence link."""

    identifier: str
    owning_phase: int
    implementation_artifact: str
    implementation_symbol: str
    node_id: str


@dataclass(frozen=True, kw_only=True, slots=True)
class AcceptanceManifest:
    """Store the immutable Phase 0 patch acceptance inventory."""

    path: Path
    current_phase: int
    nodes: tuple[AcceptanceNode, ...]

    def active_nodes(self, through_phase: int) -> tuple[AcceptanceNode, ...]:
        """Return active nodes that exist no later than one phase."""
        return tuple(
            node
            for node in self.nodes
            if node.lifecycle == "active"
            and node.active_from_phase <= through_phase
        )


def repository_root() -> Path:
    """Return the checkout root that owns this verifier."""
    return Path(__file__).resolve().parents[1]


def fixture_root() -> Path:
    """Return the tracked dormant patch fixture directory."""
    return repository_root() / "tests" / "fixtures" / _FEATURE


def default_manifest_path() -> Path:
    """Return the default tracked patch acceptance manifest path."""
    return fixture_root() / "acceptance_manifest.json"


def load_manifest(path: Path) -> AcceptanceManifest:
    """Load and validate the lifecycle-aware patch acceptance inventory."""
    payload = _load_mapping(path, "acceptance manifest")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "current_phase",
            "nodes",
            "manifest_sha256",
        },
        "acceptance manifest",
    )
    _header(payload, "acceptance manifest")
    current_phase = _phase(payload.get("current_phase"), "current_phase")
    if current_phase != _CURRENT_PHASE:
        raise PatchAcceptanceError("patch acceptance phase is not frozen")
    raw_nodes = object_list(payload.get("nodes"), "acceptance nodes")
    if not raw_nodes:
        raise PatchAcceptanceError("acceptance node inventory is empty")
    nodes = tuple(_acceptance_node(raw, current_phase) for raw in raw_nodes)
    _unique((node.identifier for node in nodes), "acceptance node ID")
    _unique((node.node_id for node in nodes), "pytest node ID")
    if not any(node.lifecycle == "active" for node in nodes):
        raise PatchAcceptanceError(
            "acceptance node inventory has no active nodes"
        )
    canonical = {
        key: value
        for key, value in payload.items()
        if key != "manifest_sha256"
    }
    if payload.get("manifest_sha256") != canonical_sha256(canonical):
        raise PatchAcceptanceError("acceptance manifest digest is invalid")
    return AcceptanceManifest(
        path=path, current_phase=current_phase, nodes=nodes
    )


def _validate_acceptance_history(
    path: Path,
    manifest: AcceptanceManifest,
    root: Path,
) -> None:
    """Require active acceptance history to be sealed and monotonic."""
    payload = _load_mapping(path, "acceptance history")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "snapshot",
            "replacements",
            "history_sha256",
        },
        "acceptance history",
    )
    _header(payload, "acceptance history")
    snapshot = mapping(payload.get("snapshot"), "acceptance history snapshot")
    _exact_keys(
        snapshot,
        {"nodes", "snapshot_sha256"},
        "acceptance history snapshot",
    )
    nodes = object_list(snapshot.get("nodes"), "acceptance history nodes")
    historical: dict[str, Mapping[str, object]] = {}
    for raw in nodes:
        snapshot_node = _history_snapshot_node(raw, root)
        identifier = _identifier(
            snapshot_node.get("id"), "acceptance history node ID"
        )
        if identifier in historical:
            raise PatchAcceptanceError(
                "acceptance history node IDs are duplicated"
            )
        historical[identifier] = snapshot_node
    if not historical:
        raise PatchAcceptanceError("acceptance history snapshot is empty")
    snapshot_digest = canonical_sha256(nodes)
    if snapshot.get("snapshot_sha256") != snapshot_digest:
        raise PatchAcceptanceError(
            "acceptance history snapshot digest is invalid"
        )
    if snapshot_digest != _PINNED_ACCEPTANCE_HISTORY_SNAPSHOT_SHA256:
        raise PatchAcceptanceError("acceptance history snapshot is not pinned")
    replacements: dict[str, Mapping[str, object]] = {}
    for raw in object_list(
        payload.get("replacements"), "acceptance history replacements"
    ):
        replacement_entry = mapping(raw, "acceptance history replacement")
        _exact_keys(
            replacement_entry,
            {
                "old",
                "new",
                "review_round",
                "reviewer",
                "rationale",
            },
            "acceptance history replacement",
        )
        old = _history_snapshot_node(replacement_entry.get("old"), root)
        old_id = _identifier(old.get("id"), "replacement old ID")
        if old_id in replacements or old_id not in historical:
            raise PatchAcceptanceError(
                "acceptance history replacement is invalid"
            )
        if old != historical[old_id]:
            raise PatchAcceptanceError(
                "acceptance history replacement source drifted"
            )
        new = _history_snapshot_node(replacement_entry.get("new"), root)
        if _phase(
            new.get("active_from_phase"), "replacement new phase"
        ) < _phase(old.get("active_from_phase"), "replacement old phase"):
            raise PatchAcceptanceError("acceptance history phase regressed")
        round_number = replacement_entry.get("review_round")
        if type(round_number) is not int or not 1 <= round_number <= 5:
            raise PatchAcceptanceError(
                "acceptance history review round is invalid"
            )
        _string(
            replacement_entry.get("reviewer"),
            "acceptance history reviewer",
        )
        _string(
            replacement_entry.get("rationale"),
            "acceptance history rationale",
        )
        replacements[old_id] = new
    current = {
        node.identifier: _acceptance_node_snapshot(node, root)
        for node in manifest.active_nodes(manifest.current_phase)
    }
    for old_id, old in historical.items():
        current_node = current.get(old_id)
        if current_node == old:
            if old_id in replacements:
                raise PatchAcceptanceError(
                    "unchanged acceptance history has a replacement record"
                )
            continue
        replacement = replacements.get(old_id)
        if replacement is None:
            raise PatchAcceptanceError(
                "acceptance history semantic change is unreviewed"
            )
        new_id = _identifier(replacement.get("id"), "replacement new ID")
        if current.get(new_id) != replacement:
            raise PatchAcceptanceError(
                "acceptance history replacement is not active"
            )
    expected_current_ids = (set(historical) - set(replacements)) | {
        _identifier(replacement.get("id"), "replacement new ID")
        for replacement in replacements.values()
    }
    if set(current) != expected_current_ids:
        raise PatchAcceptanceError(
            "acceptance history semantic change is unreviewed"
        )
    canonical = {
        key: value for key, value in payload.items() if key != "history_sha256"
    }
    if payload.get("history_sha256") != canonical_sha256(canonical):
        raise PatchAcceptanceError("acceptance history digest is invalid")


def load_phase0_contracts(
    fixture_directory: Path | None = None,
    *,
    repo_root: Path | None = None,
) -> AcceptanceManifest:
    """Load the self-contained Phase 0 fixtures without executing pytest."""
    root = (repo_root or repository_root()).resolve()
    fixtures = (fixture_directory or fixture_root()).resolve()
    _require_fixture_bundle(fixtures)
    manifest = load_manifest(fixtures / "acceptance_manifest.json")
    _validate_acceptance_history(
        fixtures / "acceptance_history.json", manifest, root
    )
    requirements = _validate_requirements(
        fixtures / "requirements_traceability.json", manifest, root
    )
    _validate_contract_decisions(
        fixtures / "contract_decisions.json",
        root,
        fixtures / "source_symbols.json",
    )
    _validate_failure_matrix(
        fixtures / "failure_matrix.json", manifest, requirements
    )
    target_ids = _validate_target_conformance(
        fixtures / "target_conformance.json", manifest
    )
    _validate_surface_conformance(
        fixtures / "surface_conformance.json", manifest, target_ids
    )
    _validate_source_symbols(fixtures / "source_symbols.json", root)
    _validate_type_contract(fixtures / "type_contract_manifest.json")
    _validate_goldens(fixtures / "goldens.json", manifest, requirements)
    _validate_threat_model(
        fixtures / "threat_model.json", manifest, requirements
    )
    _validate_baseline_evidence(
        fixtures / "baseline_evidence.json",
        fixtures / "contract_decisions.json",
        root,
    )
    _validate_phase_evidence(fixtures / "phase_evidence.json", manifest, root)
    verify_source_artifact_reads(root)
    return manifest


def verify_acceptance(
    manifest_path: Path | None = None,
    *,
    repo_root: Path | None = None,
    through_phase: int,
    fixture_directory: Path | None = None,
) -> AcceptanceManifest:
    """Validate contracts and execute active Phase 0 nodes exactly once."""
    root = (repo_root or repository_root()).resolve()
    if through_phase != _CURRENT_PHASE:
        raise PatchAcceptanceError("patch acceptance phase is not implemented")
    path = manifest_path or default_manifest_path()
    fixtures = fixture_directory or path.parent
    manifest = load_phase0_contracts(fixtures, repo_root=root)
    if path.resolve() != manifest.path.resolve():
        manifest = load_manifest(path)
    nodes = manifest.active_nodes(through_phase)
    if not nodes:
        raise PatchAcceptanceError("selected patch phase has no active nodes")
    try:
        with TemporaryDirectory(
            prefix="avalan-patch-acceptance-"
        ) as temporary:
            execute_pytest_nodes(
                root,
                tuple(node.node_id for node in nodes),
                junit_path=Path(temporary) / "pytest.xml",
            )
    except ContractGateError as exc:
        raise PatchAcceptanceError(str(exc)) from exc
    return manifest


def verify_source_artifact_reads(root: Path) -> None:
    """Reject source or tests that read ignored patch design artifacts."""
    for path in _python_paths(root):
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            tree = parse_python(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise PatchAcceptanceError(
                f"cannot inspect patch source guard path: {path}"
            ) from exc
        aliases = _artifact_aliases(tree)
        values = _artifact_path_values(tree, aliases)
        reader_aliases = _artifact_reader_aliases(tree, aliases, values)
        for call in (node for node in walk(tree) if isinstance(node, Call)):
            artifact = _read_artifact(call, aliases, values, reader_aliases)
            if artifact is None:
                continue
            raise PatchAcceptanceError(
                "tracked source reads ignored patch design artifact: "
                f"{path.relative_to(root)}:{artifact}"
            )


def verify_source_artifact_runtime_open(
    operation: Callable[[], object],
) -> None:
    """Reject one actual guarded open of an ignored patch design artifact."""
    token = _SOURCE_ARTIFACT_AUDIT_ENABLED.set(True)
    try:
        operation()
    finally:
        _SOURCE_ARTIFACT_AUDIT_ENABLED.reset(token)


def _acceptance_node(raw: object, current_phase: int) -> AcceptanceNode:
    item = mapping(raw, "acceptance node")
    _exact_keys(
        item,
        {
            "id",
            "category",
            "lifecycle",
            "active_from_phase",
            "requirement_ids",
            "node_id",
            "surface",
            "context",
            "platform",
            "operation",
            "authority",
            "commit_boundary",
            "evidence_class",
        },
        "acceptance node",
    )
    identifier = _identifier(item.get("id"), "acceptance node ID")
    category = _string(item.get("category"), "acceptance node category")
    if category not in _NODE_CATEGORIES:
        raise PatchAcceptanceError("acceptance node category is invalid")
    lifecycle = _string(item.get("lifecycle"), "acceptance node lifecycle")
    phase = _phase(item.get("active_from_phase"), "acceptance node phase")
    expected_lifecycle = "active" if phase <= current_phase else "planned"
    if lifecycle != expected_lifecycle:
        raise PatchAcceptanceError(
            "acceptance node lifecycle is prematurely active or inactive"
        )
    requirement_ids = _identifier_list(
        item.get("requirement_ids"), "acceptance requirement IDs"
    )
    if not requirement_ids:
        raise PatchAcceptanceError("acceptance node has no requirement owner")
    _test_node(item.get("node_id"))
    semantics = {
        label: _string(item.get(label), f"acceptance node {label}")
        for label in (
            "surface",
            "context",
            "platform",
            "operation",
            "authority",
            "commit_boundary",
            "evidence_class",
        )
    }
    return AcceptanceNode(
        identifier=identifier,
        category=category,
        lifecycle=lifecycle,
        active_from_phase=phase,
        requirement_ids=requirement_ids,
        node_id=_test_node(item.get("node_id")),
        surface=semantics["surface"],
        context=semantics["context"],
        platform=semantics["platform"],
        operation=semantics["operation"],
        permission_label=semantics["authority"],
        commit_boundary=semantics["commit_boundary"],
        evidence_class=semantics["evidence_class"],
    )


def _history_snapshot_node(value: object, root: Path) -> Mapping[str, object]:
    """Return one complete, current executable acceptance snapshot node."""
    raw = mapping(value, "acceptance history node")
    _exact_keys(
        raw,
        {
            "id",
            "requirement_ids",
            "category",
            "surface",
            "context",
            "platform",
            "operation",
            "authority",
            "commit_boundary",
            "evidence_class",
            "node_id",
            "lifecycle",
            "active_from_phase",
            "executable_sha256",
        },
        "acceptance history node",
    )
    executable_digest = _sha256(
        raw.get("executable_sha256"), "acceptance history executable digest"
    )
    node = _acceptance_node(
        {key: item for key, item in raw.items() if key != "executable_sha256"},
        _CURRENT_PHASE,
    )
    if node.lifecycle != "active":
        raise PatchAcceptanceError(
            "acceptance history node must retain active lifecycle"
        )
    snapshot = _acceptance_node_snapshot(node, root)
    if snapshot["executable_sha256"] != executable_digest:
        raise PatchAcceptanceError(
            "acceptance history executable digest drifted"
        )
    return snapshot


def _acceptance_node_snapshot(
    node: AcceptanceNode, root: Path
) -> Mapping[str, object]:
    """Return all frozen node semantics and its executable source digest."""
    return {
        "id": node.identifier,
        "requirement_ids": list(node.requirement_ids),
        "category": node.category,
        "surface": node.surface,
        "context": node.context,
        "platform": node.platform,
        "operation": node.operation,
        "authority": node.permission_label,
        "commit_boundary": node.commit_boundary,
        "evidence_class": node.evidence_class,
        "node_id": node.node_id,
        "lifecycle": node.lifecycle,
        "active_from_phase": node.active_from_phase,
        "executable_sha256": _acceptance_node_executable_digest(node, root),
    }


def _acceptance_node_executable_digest(
    node: AcceptanceNode, root: Path
) -> str:
    """Return the source digest for the exact executable owned by one node."""
    path_value, symbol = node.node_id.split("::", maxsplit=1)
    path = root / path_value
    try:
        source = path.read_text(encoding="utf-8")
        tree = parse_python(source, filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise PatchAcceptanceError(
            "acceptance history executable is unavailable"
        ) from exc
    matches = tuple(
        candidate
        for candidate in tree.body
        if isinstance(candidate, (AsyncFunctionDef, FunctionDef))
        and candidate.name == symbol
    )
    if len(matches) != 1:
        raise PatchAcceptanceError(
            "acceptance history executable node is not exact"
        )
    segment = get_source_segment(source, matches[0])
    if segment is None:
        raise PatchAcceptanceError(
            "acceptance history executable source is unavailable"
        )
    return sha256(segment.encode("utf-8")).hexdigest()


def _validate_requirements(
    path: Path,
    manifest: AcceptanceManifest,
    root: Path,
) -> dict[str, RequirementOwnership]:
    payload = _load_mapping(path, "requirements traceability")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "source_sha256",
            "source_line_count",
            "expected_requirement_count",
            "record_layout",
            "requirements",
            "catalog_sha256",
        },
        "requirements traceability",
    )
    _header(payload, "requirements traceability", schema_version=2)
    _sha256(payload.get("source_sha256"), "requirements source digest")
    line_count = _line(
        payload.get("source_line_count"), "requirements source line count"
    )
    if payload.get("record_layout") != list(_REQUIREMENT_RECORD_LAYOUT):
        raise PatchAcceptanceError("requirements record layout is invalid")
    raw_requirements = object_list(payload.get("requirements"), "requirements")
    if not raw_requirements:
        raise PatchAcceptanceError("requirements traceability is empty")
    expected_count = payload.get("expected_requirement_count")
    if type(expected_count) is not int or expected_count != len(
        raw_requirements
    ):
        raise PatchAcceptanceError(
            "requirements traceability count is invalid"
        )
    identifiers: list[str] = []
    ownership_by_identifier: dict[str, RequirementOwnership] = {}
    spans: set[tuple[str, int, int]] = set()
    source_sections: set[str] = set()
    appendix_sections: set[str] = set()
    for index, raw in enumerate(raw_requirements, start=1):
        values = object_list(raw, "requirement")
        if len(values) != len(_REQUIREMENT_RECORD_LAYOUT):
            raise PatchAcceptanceError("requirement record has invalid length")
        identifier = _identifier(values[0], "requirement ID")
        if identifier in ownership_by_identifier:
            raise PatchAcceptanceError("duplicate requirement ID")
        expected_identifier = f"PATCH-R-{index:04d}"
        if identifier != expected_identifier:
            raise PatchAcceptanceError("requirement IDs are not contiguous")
        identifiers.append(identifier)
        source_section = _string(values[1], "requirement source section")
        if _SOURCE_SECTION_PATTERN.fullmatch(source_section) is None:
            raise PatchAcceptanceError("requirement source section is invalid")
        start = _line(values[2], "requirement start line")
        end = _line(values[3], "requirement end line")
        if end < start or end > line_count:
            raise PatchAcceptanceError(
                "requirement source line span is invalid"
            )
        span = (source_section, start, end)
        if span in spans:
            raise PatchAcceptanceError("duplicate requirement source span")
        spans.add(span)
        source_sections.add(source_section.split(".", 1)[0])
        if source_section.startswith("A."):
            appendix_sections.add(source_section)
        _string(values[4], "requirement statement")
        modality = _string(values[5], "requirement modality")
        if modality not in _REQUIREMENT_MODALITIES:
            raise PatchAcceptanceError("requirement modality is invalid")
        phase = _phase(values[6], "requirement owning phase")
        disposition = _string(
            values[7],
            "requirement optional feature disposition",
        )
        if disposition not in {
            "required",
            "optional",
            "optional_inactive",
        }:
            raise PatchAcceptanceError(
                "requirement optional feature disposition is invalid"
            )
        source_kind = _string(values[8], "requirement source kind")
        if source_kind not in _REQUIREMENT_SOURCE_KINDS:
            raise PatchAcceptanceError("requirement source kind is invalid")
        if source_kind == "acceptance" and modality != "ACCEPTANCE":
            raise PatchAcceptanceError(
                "acceptance requirement modality is invalid"
            )
        if source_kind == "appendix" and modality != "APPENDIX":
            raise PatchAcceptanceError(
                "appendix requirement modality is invalid"
            )
        if source_kind == "documentation" and modality != "DOCUMENTATION":
            raise PatchAcceptanceError(
                "documentation requirement modality is invalid"
            )
        if source_kind == "locked_decision" and modality != "DECISION":
            raise PatchAcceptanceError("locked decision modality is invalid")
        if source_kind == "non_goal" and modality != "NON_GOAL":
            raise PatchAcceptanceError(
                "non-goal requirement modality is invalid"
            )
        artifact = str(
            _safe_artifact_path(
                values[9], "requirement implementation artifact"
            )
        )
        symbol = _symbol(values[10], "requirement implementation symbol")
        node_id = _test_node(values[11])
        ownership_by_identifier[identifier] = RequirementOwnership(
            identifier=identifier,
            owning_phase=phase,
            implementation_artifact=artifact,
            implementation_symbol=symbol,
            node_id=node_id,
        )
        _string(values[12], "requirement documentation owner")
        _string(values[13], "requirement evidence class")
    missing_sections = _REQUIRED_SOURCE_SECTIONS - source_sections
    missing_appendix = _REQUIRED_APPENDIX_SECTIONS - appendix_sections
    if missing_sections or missing_appendix:
        missing = sorted(missing_sections | missing_appendix)
        raise PatchAcceptanceError(
            f"required source areas are missing: {', '.join(missing)}"
        )
    nodes_by_requirement: dict[str, set[str]] = {
        identifier: set() for identifier in ownership_by_identifier
    }
    manifest_nodes = {node.node_id: node for node in manifest.nodes}
    for node in manifest.nodes:
        for identifier in node.requirement_ids:
            if identifier not in nodes_by_requirement:
                raise PatchAcceptanceError(
                    "acceptance node references an unknown requirement"
                )
            nodes_by_requirement[identifier].add(node.node_id)
    active_node_requirements: dict[str, list[RequirementOwnership]] = {}
    for identifier, node_ids in nodes_by_requirement.items():
        ownership = ownership_by_identifier[identifier]
        if node_ids != {ownership.node_id}:
            raise PatchAcceptanceError(
                "requirement is unowned or differs from acceptance evidence: "
                f"{identifier}"
            )
        owner = manifest_nodes.get(ownership.node_id)
        if owner is None or owner.active_from_phase != ownership.owning_phase:
            raise PatchAcceptanceError(
                "requirement owner phase differs from acceptance evidence: "
                f"{identifier}"
            )
        if ownership.owning_phase == _CURRENT_PHASE:
            if owner.lifecycle != "active":
                raise PatchAcceptanceError(
                    "active requirement has planned evidence"
                )
            active_node_requirements.setdefault(owner.node_id, []).append(
                ownership
            )
            _validate_active_requirement_artifact(ownership, root)
        else:
            if owner.lifecycle != "planned":
                raise PatchAcceptanceError(
                    "planned requirement has active evidence"
                )
            _validate_planned_requirement_artifact(ownership)
    for node_id, owned in active_node_requirements.items():
        if len(owned) > _MAX_ACTIVE_REQUIREMENTS_PER_NODE:
            raise PatchAcceptanceError(
                f"active acceptance node owns too many requirements: {node_id}"
            )
        _validate_active_node_semantic_evidence(root, node_id, owned)
    canonical = {
        "record_layout": payload["record_layout"],
        "requirements": raw_requirements,
    }
    if payload.get("catalog_sha256") != canonical_sha256(canonical):
        raise PatchAcceptanceError(
            "requirements traceability digest is invalid"
        )
    return ownership_by_identifier


def _validate_active_requirement_artifact(
    ownership: RequirementOwnership, root: Path
) -> None:
    """Require a Phase 0 requirement to bind one executable local artifact."""
    relative = ownership.implementation_artifact
    if not relative.startswith(_ACTIVE_IMPLEMENTATION_ROOT):
        raise PatchAcceptanceError(
            "active requirement implementation artifact is not executable"
        )
    source = root / relative
    if source.is_symlink() or not source.is_file() or source.suffix != ".py":
        raise PatchAcceptanceError("active requirement artifact is missing")
    if not _python_symbol_exists(source, (ownership.implementation_symbol,)):
        raise PatchAcceptanceError(
            "active requirement implementation symbol is missing"
        )


def _validate_planned_requirement_artifact(
    ownership: RequirementOwnership,
) -> None:
    """Require future requirements to name an exact production target."""
    relative = ownership.implementation_artifact
    if not relative.startswith(_PLANNED_IMPLEMENTATION_ROOTS):
        raise PatchAcceptanceError(
            "planned requirement artifact is not a future production artifact"
        )
    if relative.startswith("tests/") or relative.endswith("/"):
        raise PatchAcceptanceError("planned requirement artifact is not exact")


def _validate_active_node_semantic_evidence(
    root: Path,
    node_id: str,
    ownership: list[RequirementOwnership],
) -> None:
    """Require each Phase 0 node to execute its named contract symbols."""
    function = _test_node_function(root, node_id)
    if _function_calls_name(function, "load_phase0_contracts"):
        raise PatchAcceptanceError(
            "active requirement evidence cannot be a generic bundle load"
        )
    names = {
        value.id for value in walk(function) if isinstance(value, Name)
    } | {
        value.attr for value in walk(function) if isinstance(value, Attribute)
    }
    missing = sorted(
        item.implementation_symbol
        for item in ownership
        if item.implementation_symbol not in names
    )
    if missing:
        raise PatchAcceptanceError(
            "active requirement evidence does not execute its implementation "
            f"symbol: {', '.join(missing)}"
        )


def _test_node_function(
    root: Path, node_id: str
) -> FunctionDef | AsyncFunctionDef:
    """Return the exact non-parameterized test function named by one node."""
    parts = node_id.split("::")
    path = root / parts[0]
    if path.is_symlink() or not path.is_file():
        raise PatchAcceptanceError("active requirement test node is missing")
    try:
        tree = parse_python(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise PatchAcceptanceError(
            "cannot inspect active requirement evidence node"
        ) from exc
    selector = parts[1:]
    if len(selector) == 1:
        candidates = tuple(
            value
            for value in tree.body
            if isinstance(value, (FunctionDef, AsyncFunctionDef))
            and value.name == selector[0]
        )
    elif len(selector) == 2:
        classes = tuple(
            value
            for value in tree.body
            if isinstance(value, ClassDef) and value.name == selector[0]
        )
        candidates = tuple(
            value
            for item in classes
            for value in item.body
            if isinstance(value, (FunctionDef, AsyncFunctionDef))
            and value.name == selector[1]
        )
    else:
        candidates = ()
    if len(candidates) != 1:
        raise PatchAcceptanceError("active requirement test node is not exact")
    return candidates[0]


def _function_calls_name(
    function: FunctionDef | AsyncFunctionDef, name: str
) -> bool:
    """Return whether one test function directly invokes one named helper."""
    for value in walk(function):
        if not isinstance(value, Call):
            continue
        target = value.func
        if isinstance(target, Name) and target.id == name:
            return True
        if isinstance(target, Attribute) and target.attr == name:
            return True
    return False


def _validate_failure_matrix(
    path: Path,
    manifest: AcceptanceManifest,
    requirement_ids: Mapping[str, RequirementOwnership],
) -> None:
    payload = _load_mapping(path, "failure matrix")
    _exact_keys(
        payload,
        {"schema_version", "feature", "cells", "matrix_sha256"},
        "failure matrix",
    )
    _header(payload, "failure matrix")
    cells = object_list(payload.get("cells"), "failure cells")
    if not cells:
        raise PatchAcceptanceError("failure matrix is empty")
    identifiers: list[str] = []
    boundaries: list[str] = []
    node_ids = {node.node_id: node for node in manifest.nodes}
    for raw in cells:
        item = mapping(raw, "failure cell")
        _exact_keys(
            item,
            {
                "id",
                "lifecycle",
                "active_from_phase",
                "boundary",
                "surface",
                "requirement_id",
                "evidence_node_id",
                "commit_started",
                "expected_inspection_count",
                "expected_workspace_write_count",
                "expected_dispatch_count",
                "expected_approval_count",
                "expected_commit_count",
                "per_step_states",
                "per_lineage_states",
                "artifact_state",
                "requested_effect_fact",
                "workspace_change_fact",
                "event_sequence",
                "pending_behavior",
                "retryability",
                "target_inspection_count",
                "workspace_oracle_equal",
                "target_workspace_mutation_count",
                "public_projection",
            },
            "failure cell",
        )
        identifier = _identifier(item.get("id"), "failure cell ID")
        identifiers.append(identifier)
        boundary = _string(item.get("boundary"), "failure boundary")
        boundaries.append(boundary)
        if boundary not in _FAILURE_BOUNDARY_CATALOG:
            raise PatchAcceptanceError("failure boundary is not frozen")
        semantic = _FAILURE_SEMANTICS[boundary]
        phase = _phase(item.get("active_from_phase"), "failure cell phase")
        lifecycle = _string(item.get("lifecycle"), "failure cell lifecycle")
        expected = "active" if phase <= manifest.current_phase else "planned"
        if lifecycle != expected:
            raise PatchAcceptanceError("failure cell lifecycle is invalid")
        _string(item.get("surface"), "failure surface")
        if (
            _identifier(item.get("requirement_id"), "failure requirement ID")
            not in requirement_ids
        ):
            raise PatchAcceptanceError(
                "failure cell references unknown requirement"
            )
        evidence_node_id = _test_node(item.get("evidence_node_id"))
        node = node_ids.get(evidence_node_id)
        if (
            node is None
            or node.lifecycle != "active"
            or node.active_from_phase > phase
        ):
            raise PatchAcceptanceError(
                "failure cell evidence node is absent or inactive"
            )
        commit_started = item.get("commit_started")
        if type(commit_started) is not bool:
            raise PatchAcceptanceError(
                "failure commit_started must be boolean"
            )
        assert isinstance(commit_started, bool)
        counts = {
            field: _nonnegative_int(item.get(field), f"failure {field}")
            for field in (
                "expected_inspection_count",
                "expected_workspace_write_count",
                "expected_dispatch_count",
                "expected_approval_count",
                "expected_commit_count",
                "target_inspection_count",
                "target_workspace_mutation_count",
            )
        }
        if (
            counts["expected_inspection_count"]
            != counts["target_inspection_count"]
        ):
            raise PatchAcceptanceError(
                "failure target inspection count drifted"
            )
        if (
            counts["expected_workspace_write_count"]
            != counts["target_workspace_mutation_count"]
        ):
            raise PatchAcceptanceError("failure workspace write count drifted")
        expected_counts = {
            "expected_inspection_count": semantic.counts[0],
            "expected_workspace_write_count": semantic.counts[1],
            "expected_dispatch_count": semantic.counts[2],
            "expected_approval_count": semantic.counts[3],
            "expected_commit_count": semantic.counts[4],
            "target_inspection_count": semantic.counts[0],
            "target_workspace_mutation_count": semantic.counts[1],
        }
        if counts != expected_counts:
            raise PatchAcceptanceError("failure expected counts drifted")
        _validate_failure_state_vector(
            item.get("per_step_states"),
            "step",
            _FAILURE_STEP_STATES,
        )
        _validate_failure_state_vector(
            item.get("per_lineage_states"),
            "lineage",
            _FAILURE_LINEAGE_STATES,
        )
        artifact_state = _string(
            item.get("artifact_state"), "failure artifact state"
        )
        if artifact_state not in _FAILURE_ARTIFACT_STATES:
            raise PatchAcceptanceError("failure artifact state is invalid")
        effect_fact = _string(
            item.get("requested_effect_fact"), "failure effect fact"
        )
        if effect_fact not in _FAILURE_EFFECT_FACTS:
            raise PatchAcceptanceError(
                "failure requested effect fact is invalid"
            )
        workspace_fact = _string(
            item.get("workspace_change_fact"), "failure workspace fact"
        )
        if workspace_fact not in _FAILURE_WORKSPACE_FACTS:
            raise PatchAcceptanceError(
                "failure workspace change fact is invalid"
            )
        events = tuple(
            _string(value, "failure event")
            for value in object_list(
                item.get("event_sequence"), "failure event sequence"
            )
        )
        if not events or any(value not in _FAILURE_EVENTS for value in events):
            raise PatchAcceptanceError("failure event sequence is invalid")
        if events != tuple(dict.fromkeys(events)):
            raise PatchAcceptanceError("failure event sequence is not exact")
        if events != semantic.events:
            raise PatchAcceptanceError("failure event sequence drifted")
        pending = _string(
            item.get("pending_behavior"), "failure pending behavior"
        )
        if pending not in _FAILURE_PENDING_BEHAVIORS:
            raise PatchAcceptanceError("failure pending behavior is invalid")
        retryability = _string(
            item.get("retryability"), "failure retryability"
        )
        if retryability not in _FAILURE_RETRYABILITIES:
            raise PatchAcceptanceError("failure retryability is invalid")
        if type(item.get("workspace_oracle_equal")) is not bool:
            raise PatchAcceptanceError(
                "failure workspace oracle must be boolean"
            )
        projection = _string(
            item.get("public_projection"), "failure public projection"
        )
        if projection not in _FAILURE_PROJECTIONS:
            raise PatchAcceptanceError("failure public projection is invalid")
        if not item["commit_started"] and (
            not item["workspace_oracle_equal"]
            or counts["expected_workspace_write_count"] != 0
            or counts["target_workspace_mutation_count"] != 0
            or workspace_fact != "unchanged"
            or effect_fact != "not_committed"
        ):
            raise PatchAcceptanceError(
                "precommit failure lacks both zero-write evidence oracles"
            )
        _validate_failure_semantic_truth(
            semantic=semantic,
            commit_started=commit_started,
            step_states=item.get("per_step_states"),
            lineage_states=item.get("per_lineage_states"),
            artifact_state=artifact_state,
            effect_fact=effect_fact,
            workspace_fact=workspace_fact,
            workspace_oracle_equal=item.get("workspace_oracle_equal"),
            pending=pending,
            retryability=retryability,
            projection=projection,
        )
    _unique(identifiers, "failure cell ID")
    _unique(boundaries, "failure boundary")
    if tuple(sorted(boundaries)) != tuple(sorted(_FAILURE_BOUNDARY_CATALOG)):
        raise PatchAcceptanceError("failure boundary catalog is incomplete")
    canonical = {
        key: value for key, value in payload.items() if key != "matrix_sha256"
    }
    if payload.get("matrix_sha256") != canonical_sha256(canonical):
        raise PatchAcceptanceError("failure matrix digest is invalid")


def _validate_failure_semantic_truth(
    *,
    semantic: _FailureSemantic,
    commit_started: bool,
    step_states: object,
    lineage_states: object,
    artifact_state: str,
    effect_fact: str,
    workspace_fact: str,
    workspace_oracle_equal: object,
    pending: str,
    retryability: str,
    projection: str,
) -> None:
    """Reject a boundary vector that cannot describe its declared outcome."""
    expected_step = (("PATCH-STEP-001", semantic.step_state),)
    expected_lineage = (("PATCH-LINEAGE-001", semantic.lineage_state),)
    actual_step = tuple(
        (
            _string(item.get("id"), "failure step ID"),
            _string(item.get("state"), "failure step state"),
        )
        for raw in object_list(step_states, "failure step states")
        for item in (mapping(raw, "failure step state"),)
    )
    actual_lineage = tuple(
        (
            _string(item.get("id"), "failure lineage ID"),
            _string(item.get("state"), "failure lineage state"),
        )
        for raw in object_list(lineage_states, "failure lineage states")
        for item in (mapping(raw, "failure lineage state"),)
    )
    if (
        commit_started != semantic.commit_started
        or actual_step != expected_step
        or actual_lineage != expected_lineage
        or artifact_state != semantic.artifact_state
        or effect_fact != semantic.requested_effect_fact
        or workspace_fact != semantic.workspace_change_fact
        or workspace_oracle_equal is not semantic.workspace_oracle_equal
        or pending != semantic.pending_behavior
        or retryability != semantic.retryability
        or projection != semantic.public_projection
    ):
        raise PatchAcceptanceError("failure semantic truth drifted")
    if (
        pending.startswith("pending_")
        and semantic.events[-1] == "request_completed"
    ):
        raise PatchAcceptanceError("pending failure emits a terminal event")
    if not commit_started and (
        semantic.step_state != "not_committed"
        or semantic.lineage_state != "not_committed"
        or semantic.requested_effect_fact != "not_committed"
        or semantic.workspace_change_fact != "unchanged"
    ):
        raise PatchAcceptanceError("precommit failure has postcommit truth")
    if (
        commit_started
        and semantic.step_state == "not_committed"
        and semantic.public_projection != "patch_commit_failed"
    ):
        raise PatchAcceptanceError(
            "postcommit not_committed projection is invalid"
        )


def _validate_failure_state_vector(
    value: object,
    kind: str,
    permitted_states: frozenset[str],
) -> None:
    """Require one non-empty, sorted, closed requested-effect state vector."""
    identifiers: list[str] = []
    for raw in object_list(value, f"failure {kind} states"):
        item = mapping(raw, f"failure {kind} state")
        _exact_keys(item, {"id", "state"}, f"failure {kind} state")
        identifiers.append(_identifier(item.get("id"), f"failure {kind} ID"))
        if _string(item.get("state"), f"failure {kind} state") not in (
            permitted_states
        ):
            raise PatchAcceptanceError(f"failure {kind} state is invalid")
    if not identifiers or identifiers != sorted(identifiers):
        raise PatchAcceptanceError(f"failure {kind} state vector is invalid")
    _unique(identifiers, f"failure {kind} state ID")


def _validate_target_conformance(
    path: Path,
    manifest: AcceptanceManifest,
) -> frozenset[str]:
    payload = _load_mapping(path, "target conformance")
    _exact_keys(
        payload,
        {"schema_version", "feature", "profiles", "manifest_sha256"},
        "target conformance",
    )
    _header(payload, "target conformance")
    profiles = object_list(payload.get("profiles"), "target profiles")
    identifiers: list[str] = []
    contexts: list[str] = []
    active_nodes = {
        node.node_id for node in manifest.active_nodes(_CURRENT_PHASE)
    }
    for raw in profiles:
        item = mapping(raw, "target profile")
        _exact_keys(
            item,
            {
                "id",
                "context",
                "platform",
                "filesystem",
                "lifecycle",
                "active_from_phase",
                "advertised",
                "mutation_worker_protocol",
                "metadata_profile",
                "primitive",
                "lease",
                "coordinator",
                "persistence",
                "evidence_node_id",
            },
            "target profile",
        )
        identifiers.append(_identifier(item.get("id"), "target profile ID"))
        context = _string(item.get("context"), "target context")
        contexts.append(context)
        if context not in _TARGET_CONTEXTS:
            raise PatchAcceptanceError("target context is invalid")
        _string(item.get("platform"), "target platform")
        _string(item.get("filesystem"), "target filesystem")
        phase = _phase(item.get("active_from_phase"), "target profile phase")
        if _string(item.get("lifecycle"), "target profile lifecycle") != (
            "active" if phase <= manifest.current_phase else "planned"
        ):
            raise PatchAcceptanceError("target profile lifecycle is invalid")
        advertised = item.get("advertised")
        if type(advertised) is not bool:
            raise PatchAcceptanceError("target advertisement must be boolean")
        for field in (
            "mutation_worker_protocol",
            "metadata_profile",
            "primitive",
            "lease",
            "coordinator",
            "persistence",
        ):
            _string(item.get(field), f"target {field}")
        evidence = _test_node(item.get("evidence_node_id"))
        if evidence not in active_nodes:
            raise PatchAcceptanceError("target profile evidence is not active")
        if advertised:
            missing = tuple(
                field
                for field in (
                    "mutation_worker_protocol",
                    "metadata_profile",
                    "primitive",
                    "lease",
                    "coordinator",
                    "persistence",
                )
                if item[field] in {"none", "planned", "absent"}
            )
            if missing:
                raise PatchAcceptanceError(
                    "advertised target profile has incomplete capability "
                    "evidence"
                )
    _unique(identifiers, "target profile ID")
    if frozenset(contexts) != _TARGET_CONTEXTS or len(contexts) != len(
        _TARGET_CONTEXTS
    ):
        raise PatchAcceptanceError("target contexts must be exact and unique")
    canonical = {
        key: value
        for key, value in payload.items()
        if key != "manifest_sha256"
    }
    if payload.get("manifest_sha256") != canonical_sha256(canonical):
        raise PatchAcceptanceError("target conformance digest is invalid")
    return frozenset(identifiers)


def _validate_surface_conformance(
    path: Path,
    manifest: AcceptanceManifest,
    target_ids: frozenset[str],
) -> None:
    payload = _load_mapping(path, "surface conformance")
    _exact_keys(
        payload,
        {"schema_version", "feature", "surfaces", "manifest_sha256"},
        "surface conformance",
    )
    _header(payload, "surface conformance")
    rows = object_list(payload.get("surfaces"), "surface rows")
    seen: list[str] = []
    active_nodes = {
        node.node_id for node in manifest.active_nodes(_CURRENT_PHASE)
    }
    for raw in rows:
        item = mapping(raw, "surface row")
        _exact_keys(
            item,
            {
                "id",
                "lifecycle",
                "active_from_phase",
                "advertised",
                "externally_retryable",
                "target_profile_id",
                "required_capabilities",
                "persistence_ordering_evidence",
                "evidence_node_id",
                "pre_advertisement_negative_node_id",
            },
            "surface row",
        )
        identifier = _string(item.get("id"), "surface ID")
        seen.append(identifier)
        if identifier not in _SURFACE_IDS:
            raise PatchAcceptanceError("surface ID is invalid")
        phase = _phase(item.get("active_from_phase"), "surface phase")
        if _string(item.get("lifecycle"), "surface lifecycle") != (
            "active" if phase <= manifest.current_phase else "planned"
        ):
            raise PatchAcceptanceError("surface lifecycle is invalid")
        advertised = item.get("advertised")
        retryable = item.get("externally_retryable")
        if type(advertised) is not bool or type(retryable) is not bool:
            raise PatchAcceptanceError(
                "surface advertisement fields must be boolean"
            )
        if (
            _identifier(
                item.get("target_profile_id"), "surface target profile"
            )
            not in target_ids
        ):
            raise PatchAcceptanceError(
                "surface references unknown target profile"
            )
        capabilities = tuple(
            _string(value, "surface capability")
            for value in object_list(
                item.get("required_capabilities"), "surface capabilities"
            )
        )
        _unique(capabilities, "surface capability")
        ordering = _string(
            item.get("persistence_ordering_evidence"),
            "surface ordering evidence",
        )
        evidence = _test_node(item.get("evidence_node_id"))
        negative = _test_node(item.get("pre_advertisement_negative_node_id"))
        if evidence not in active_nodes or negative not in active_nodes:
            raise PatchAcceptanceError(
                "surface negative evidence is not active"
            )
        if advertised:
            required = {"scope", "target", "approval", "commit", "pending"}
            if not required <= set(capabilities) or not ordering.startswith(
                "active:"
            ):
                raise PatchAcceptanceError(
                    "advertised surface has incomplete capability evidence"
                )
        elif ordering.startswith("active:"):
            raise PatchAcceptanceError(
                "inactive surface cannot claim active ordering evidence"
            )
    _unique(seen, "surface ID")
    if frozenset(seen) != _SURFACE_IDS:
        raise PatchAcceptanceError("surface inventory is incomplete")
    canonical = {
        key: value
        for key, value in payload.items()
        if key != "manifest_sha256"
    }
    if payload.get("manifest_sha256") != canonical_sha256(canonical):
        raise PatchAcceptanceError("surface conformance digest is invalid")


def _validate_source_symbols(path: Path, root: Path) -> None:
    payload = _load_mapping(path, "source symbols")
    _exact_keys(
        payload,
        {"schema_version", "feature", "symbols", "inventory_sha256"},
        "source symbols",
    )
    _header(payload, "source symbols")
    symbols = object_list(payload.get("symbols"), "source symbols")
    if not symbols:
        raise PatchAcceptanceError("source symbol inventory is empty")
    identifiers: list[str] = []
    for raw in symbols:
        item = mapping(raw, "source symbol")
        _exact_keys(
            item,
            {"id", "path", "symbol", "artifact_class", "source_sha256"},
            "source symbol",
        )
        identifiers.append(_identifier(item.get("id"), "source symbol ID"))
        relative = _safe_source_path(item.get("path"))
        source = root / relative
        if source.is_symlink() or not source.is_file():
            raise PatchAcceptanceError(
                f"source symbol path is missing: {relative}"
            )
        expected_sha = _sha256(
            item.get("source_sha256"), "source symbol digest"
        )
        if _file_sha256(source) != expected_sha:
            raise PatchAcceptanceError(
                f"source symbol digest changed: {relative}"
            )
        symbol = _string(item.get("symbol"), "source symbol")
        if not _python_symbol_exists(source, tuple(symbol.split("."))):
            raise PatchAcceptanceError(
                f"source symbol is missing: {relative}:{symbol}"
            )
        _string(item.get("artifact_class"), "source artifact class")
    _unique(identifiers, "source symbol ID")
    if payload.get("inventory_sha256") != canonical_sha256(symbols):
        raise PatchAcceptanceError("source symbol inventory digest is invalid")


def _validate_type_contract(path: Path) -> None:
    try:
        manifest = load_type_manifest(path)
    except PatchTypeContractError as exc:
        raise PatchAcceptanceError(str(exc)) from exc
    active = tuple(
        fixture
        for fixture in manifest.fixtures
        if fixture.lifecycle == "active"
    )
    if not active or {fixture.kind for fixture in active} != {
        "positive",
        "negative",
    }:
        raise PatchAcceptanceError(
            "patch type contract evidence is incomplete"
        )


def _validate_goldens(
    path: Path,
    manifest: AcceptanceManifest,
    requirement_ids: Mapping[str, RequirementOwnership],
) -> None:
    payload = _load_mapping(path, "patch goldens")
    _exact_keys(
        payload,
        {"schema_version", "feature", "cases", "golden_sha256"},
        "patch goldens",
    )
    _header(payload, "patch goldens")
    cases = object_list(payload.get("cases"), "golden cases")
    categories: set[str] = set()
    for raw in cases:
        item = mapping(raw, "golden case")
        _exact_keys(
            item,
            {
                "id",
                "category",
                "lifecycle",
                "active_from_phase",
                "requirement_id",
                "node_id",
                "input_bytes_hex",
                "expected_bytes_hex",
                "expected_outcome",
                "expected_error",
            },
            "golden case",
        )
        _identifier(item.get("id"), "golden case ID")
        categories.add(_string(item.get("category"), "golden category"))
        phase = _phase(item.get("active_from_phase"), "golden phase")
        if _string(item.get("lifecycle"), "golden lifecycle") != (
            "active" if phase <= manifest.current_phase else "planned"
        ):
            raise PatchAcceptanceError("golden lifecycle is invalid")
        if (
            _identifier(item.get("requirement_id"), "golden requirement")
            not in requirement_ids
        ):
            raise PatchAcceptanceError("golden requirement is invalid")
        node = _test_node(item.get("node_id"))
        candidate = next(
            (item for item in manifest.nodes if item.node_id == node), None
        )
        if (
            candidate is None
            or candidate.lifecycle != "active"
            or candidate.active_from_phase > phase
        ):
            raise PatchAcceptanceError(
                "golden node is not active phase evidence"
            )
        _hex_bytes(item.get("input_bytes_hex"), "golden input")
        _hex_bytes(
            item.get("expected_bytes_hex"),
            "golden expected bytes",
            allow_empty=True,
        )
        _string(item.get("expected_outcome"), "golden expected outcome")
        _string(item.get("expected_error"), "golden expected error")
    if categories != _GOLDEN_CATEGORIES or len(cases) != len(categories):
        raise PatchAcceptanceError("golden category inventory is incomplete")
    if payload.get("golden_sha256") != canonical_sha256(cases):
        raise PatchAcceptanceError("golden fixture digest is invalid")


def _validate_threat_model(
    path: Path,
    manifest: AcceptanceManifest,
    requirement_ids: Mapping[str, RequirementOwnership],
) -> None:
    payload = _load_mapping(path, "threat model")
    _exact_keys(
        payload,
        {"schema_version", "feature", "threats", "model_sha256"},
        "threat model",
    )
    _header(payload, "threat model")
    threats = object_list(payload.get("threats"), "threats")
    seen: set[str] = set()
    for raw in threats:
        item = mapping(raw, "threat")
        _exact_keys(
            item,
            {
                "id",
                "lifecycle",
                "active_from_phase",
                "mitigation",
                "evidence_class",
                "requirement_id",
                "node_id",
                "setup_bytes_hex",
                "action_bytes_hex",
                "expected_bytes_hex",
                "expected_error",
                "expected_containment",
            },
            "threat",
        )
        identifier = _string(item.get("id"), "threat ID")
        seen.add(identifier)
        phase = _phase(item.get("active_from_phase"), "threat phase")
        if _string(item.get("lifecycle"), "threat lifecycle") != (
            "active" if phase <= manifest.current_phase else "planned"
        ):
            raise PatchAcceptanceError("threat state is invalid")
        _string(item.get("mitigation"), "threat mitigation")
        _string(item.get("evidence_class"), "threat evidence class")
        if (
            _identifier(item.get("requirement_id"), "threat requirement")
            not in requirement_ids
        ):
            raise PatchAcceptanceError("threat requirement is invalid")
        node = _test_node(item.get("node_id"))
        candidate = next(
            (item for item in manifest.nodes if item.node_id == node), None
        )
        if (
            candidate is None
            or candidate.lifecycle != "active"
            or candidate.active_from_phase > phase
        ):
            raise PatchAcceptanceError(
                "threat node is not active phase evidence"
            )
        _hex_bytes(item.get("setup_bytes_hex"), "threat setup")
        _hex_bytes(item.get("action_bytes_hex"), "threat action")
        _hex_bytes(item.get("expected_bytes_hex"), "threat expected bytes")
        _string(item.get("expected_error"), "threat expected error")
        _string(item.get("expected_containment"), "threat containment")
    if seen != _THREAT_IDS or len(threats) != len(seen):
        raise PatchAcceptanceError("threat model inventory is incomplete")
    if payload.get("model_sha256") != canonical_sha256(threats):
        raise PatchAcceptanceError("threat model digest is invalid")


def _validate_contract_decisions(
    path: Path, root: Path, source_symbols_path: Path
) -> None:
    """Validate frozen decisions and the complete Phase 0 baseline."""
    payload = _load_mapping(path, "contract decisions")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "source_metadata",
            "decisions",
            "contract_artifacts",
            "inventories",
            "inventory_sha256",
            "record_sha256",
        },
        "contract decisions",
    )
    _header(payload, "contract decisions")
    metadata = mapping(payload.get("source_metadata"), "decision metadata")
    _exact_keys(
        metadata, {"decision_count", "source_sha256"}, "decision metadata"
    )
    if metadata.get("decision_count") != 31:
        raise PatchAcceptanceError("frozen decision count is invalid")
    _sha256(metadata.get("source_sha256"), "decision source digest")
    decisions = object_list(payload.get("decisions"), "frozen decisions")
    if len(decisions) != 31:
        raise PatchAcceptanceError("frozen decisions are incomplete")
    expected_areas = {
        "approval_interaction": 4,
        "canonical_types": 4,
        "coordination_persistence": 6,
        "ownership_lifecycle": 4,
        "policy_disclosure_limits": 4,
        "public_provider_surfaces": 4,
        "target_protocol": 5,
    }
    area_counts: dict[str, int] = {}
    for index, raw in enumerate(decisions, start=1):
        item = mapping(raw, "frozen decision")
        _exact_keys(
            item,
            {
                "id",
                "area",
                "normalized_decision",
                "chosen_resolution",
                "alternatives",
                "security_consequences",
                "replacement_process",
                "owning_phase",
                "status",
            },
            "frozen decision",
        )
        if (
            _string(item.get("id"), "frozen decision ID")
            != f"PATCH-DEC-{index:03d}"
        ):
            raise PatchAcceptanceError(
                "frozen decision IDs are not contiguous"
            )
        area = _string(item.get("area"), "frozen decision area")
        area_counts[area] = area_counts.get(area, 0) + 1
        for field in (
            "normalized_decision",
            "chosen_resolution",
            "security_consequences",
            "replacement_process",
        ):
            _string(item.get(field), f"frozen decision {field}")
        alternatives = tuple(
            _string(value, "frozen decision alternative")
            for value in object_list(
                item.get("alternatives"), "frozen decision alternatives"
            )
        )
        if len(alternatives) < 2:
            raise PatchAcceptanceError(
                "frozen decision alternatives are incomplete"
            )
        _unique(alternatives, "frozen decision alternative")
        if (
            _phase(item.get("owning_phase"), "frozen decision owning phase")
            != 0
        ):
            raise PatchAcceptanceError("frozen decision phase is invalid")
        if item.get("status") != "frozen":
            raise PatchAcceptanceError("frozen decision status is invalid")
    if area_counts != expected_areas:
        raise PatchAcceptanceError("frozen decision areas are incomplete")
    _validate_contract_artifacts(payload.get("contract_artifacts"))
    inventories = mapping(payload.get("inventories"), "baseline inventories")
    if set(inventories) != _INVENTORY_AREAS:
        raise PatchAcceptanceError("baseline inventory areas are incomplete")
    _validate_source_inventory(inventories["src"], root)
    _validate_hashed_inventory(
        inventories["coverage_exclusions"], root, "coverage exclusion"
    )
    for area in _INVENTORY_AREAS - {"src", "coverage_exclusions"}:
        values = tuple(
            _string(value, f"baseline {area} entry")
            for value in object_list(inventories[area], f"baseline {area}")
        )
        if not values:
            raise PatchAcceptanceError(f"baseline {area} is empty")
        if tuple(sorted(values)) != values:
            raise PatchAcceptanceError(f"baseline {area} is not sorted")
        _unique(values, f"baseline {area} entry")
    _validate_symbol_inventory_link(
        source_symbols_path, inventories["src"], root
    )
    if payload.get("inventory_sha256") != canonical_sha256(inventories):
        raise PatchAcceptanceError("baseline inventory digest is invalid")
    canonical = {
        key: value for key, value in payload.items() if key != "record_sha256"
    }
    if payload.get("record_sha256") != canonical_sha256(canonical):
        raise PatchAcceptanceError(
            "contract decision record digest is invalid"
        )


def _validate_contract_artifacts(value: object) -> None:
    """Validate the closed lifecycle, limits, await, and tag artifacts."""
    artifacts = mapping(value, "frozen contract artifacts")
    _exact_keys(
        artifacts,
        {"lifecycle", "limits", "allowed_await_matrix", "closed_tags"},
        "frozen contract artifacts",
    )
    _validate_lifecycle_artifact(artifacts.get("lifecycle"))
    _validate_limit_artifact(artifacts.get("limits"))
    _validate_await_artifact(artifacts.get("allowed_await_matrix"))
    _validate_closed_tags(artifacts.get("closed_tags"))


def _validate_lifecycle_artifact(value: object) -> None:
    """Require a total finite lifecycle with one explicit terminal state."""
    lifecycle = mapping(value, "lifecycle contract")
    _exact_keys(
        lifecycle,
        {
            "owner",
            "initial",
            "terminal",
            "states",
            "transitions",
            "replacement",
        },
        "lifecycle contract",
    )
    if (
        _string(lifecycle.get("owner"), "lifecycle owner")
        != "PatchExecutionRecord"
    ):
        raise PatchAcceptanceError("lifecycle owner is not frozen")
    states = tuple(
        _string(item, "lifecycle state")
        for item in object_list(lifecycle.get("states"), "lifecycle states")
    )
    if states != _LIFECYCLE_STATES:
        raise PatchAcceptanceError("lifecycle state coverage is incomplete")
    if (
        lifecycle.get("initial") != states[0]
        or lifecycle.get("terminal") != states[-1]
    ):
        raise PatchAcceptanceError("lifecycle endpoints are invalid")
    transitions: list[tuple[str, str, str]] = []
    for raw in object_list(
        lifecycle.get("transitions"), "lifecycle transitions"
    ):
        transition = mapping(raw, "lifecycle transition")
        _exact_keys(
            transition, {"from", "to", "cause"}, "lifecycle transition"
        )
        transitions.append(
            (
                _string(transition.get("from"), "lifecycle transition source"),
                _string(transition.get("to"), "lifecycle transition target"),
                _string(transition.get("cause"), "lifecycle transition cause"),
            )
        )
    if tuple(transitions) != _LIFECYCLE_TRANSITIONS:
        raise PatchAcceptanceError("lifecycle transition table is incomplete")
    if any(source == states[-1] for source, _, _ in transitions):
        raise PatchAcceptanceError(
            "terminal lifecycle state has an outgoing transition"
        )
    if any(
        target not in states or source not in states
        for source, target, _ in transitions
    ):
        raise PatchAcceptanceError(
            "lifecycle transition references an unknown state"
        )
    if {target for _, target, _ in transitions} != set(states[1:]):
        raise PatchAcceptanceError(
            "lifecycle transition coverage is incomplete"
        )
    _validate_replacement(lifecycle.get("replacement"), "lifecycle")


def _validate_limit_artifact(value: object) -> None:
    """Require finite lower/default/upper limits and exact compositions."""
    limits = object_list(value, "finite limits")
    seen: dict[str, tuple[int, int, int, str]] = {}
    for raw in limits:
        limit = mapping(raw, "finite limit")
        _exact_keys(
            limit,
            {"id", "minimum", "default", "maximum", "composition"},
            "finite limit",
        )
        identifier = _string(limit.get("id"), "finite limit ID")
        minimum = _nonnegative_int(
            limit.get("minimum"), "finite limit minimum"
        )
        default = _nonnegative_int(
            limit.get("default"), "finite limit default"
        )
        maximum = _nonnegative_int(
            limit.get("maximum"), "finite limit maximum"
        )
        if minimum < 1 or not minimum <= default <= maximum:
            raise PatchAcceptanceError("finite limit bounds are invalid")
        seen[identifier] = (
            minimum,
            default,
            maximum,
            _string(limit.get("composition"), "finite limit composition"),
        )
    if set(seen) != set(_LIMIT_COMPOSITION):
        raise PatchAcceptanceError("finite limit coverage is incomplete")
    if any(
        seen[key][3] != composition
        for key, composition in _LIMIT_COMPOSITION.items()
    ):
        raise PatchAcceptanceError("finite limit composition is invalid")


def _validate_await_artifact(value: object) -> None:
    """Require one exact allowed depth vector for every await boundary."""
    rows = object_list(value, "allowed await matrix")
    seen: dict[str, tuple[int, ...]] = {}
    for raw in rows:
        row = mapping(raw, "allowed await row")
        _exact_keys(row, {"boundary", "depths"}, "allowed await row")
        boundary = _string(row.get("boundary"), "allowed await boundary")
        depths = mapping(row.get("depths"), "allowed await depths")
        _exact_keys(
            depths, set(_RESOURCE_DEPTH_FIELDS), "allowed await depths"
        )
        seen[boundary] = tuple(
            _nonnegative_int(depths.get(field), f"await depth {field}")
            for field in _RESOURCE_DEPTH_FIELDS
        )
    if seen != _AWAIT_MATRIX:
        raise PatchAcceptanceError("allowed await matrix is incomplete")


def _validate_closed_tags(value: object) -> None:
    """Require every authority discriminator to stay finite and exact."""
    tags = mapping(value, "closed tag catalog")
    expected = {
        "mutation_state": (
            "not_committed",
            "committed",
            "partially_committed",
            "indeterminate",
        ),
        "approval_decision": ("approve", "deny", "unavailable"),
        "approval_outcome": (
            "approved",
            "denied",
            "unavailable",
            "expired",
            "binding_mismatch",
            "replayed",
        ),
        "artifact_state": (
            "absent",
            "staged",
            "published",
            "cleaned",
            "indeterminate",
        ),
        "operation_kind": ("create", "update", "delete", "move"),
        "lifecycle_state": _LIFECYCLE_STATES,
    }
    if set(tags) != set(expected):
        raise PatchAcceptanceError("closed tag coverage is incomplete")
    for name, values in expected.items():
        observed = tuple(
            _string(item, f"closed tag {name}")
            for item in object_list(tags.get(name), f"closed tag {name}")
        )
        if observed != values:
            raise PatchAcceptanceError("closed tag values are invalid")


def _validate_replacement(value: object, label: str) -> None:
    """Require reviewed supersession semantics for a frozen artifact."""
    replacement = mapping(value, f"{label} replacement")
    _exact_keys(replacement, {"version", "requires"}, f"{label} replacement")
    if replacement.get("version") != 1:
        raise PatchAcceptanceError(f"{label} replacement version is invalid")
    required = tuple(
        _string(item, f"{label} replacement requirement")
        for item in object_list(
            replacement.get("requires"), f"{label} replacement requirements"
        )
    )
    if required != _REPLACEMENT_REQUIREMENTS:
        raise PatchAcceptanceError(
            f"{label} replacement semantics are incomplete"
        )


def _nonnegative_int(value: object, label: str) -> int:
    """Return one finite non-negative integer used by a contract artifact."""
    if type(value) is not int or value < 0:
        raise PatchAcceptanceError(f"{label} is invalid")
    return value


def _validate_source_inventory(value: object, root: Path) -> None:
    entries = object_list(value, "baseline src inventory")
    observed: list[dict[str, str]] = []
    for raw in entries:
        item = mapping(raw, "baseline src entry")
        _exact_keys(item, {"path", "source_sha256"}, "baseline src entry")
        relative = _safe_source_path(item.get("path"))
        observed.append(
            {
                "path": relative.as_posix(),
                "source_sha256": _sha256(
                    item.get("source_sha256"), "baseline src digest"
                ),
            }
        )
    if observed != sorted(observed, key=lambda item: item["path"]):
        raise PatchAcceptanceError("baseline src inventory is not sorted")
    _unique((item["path"] for item in observed), "baseline src path")
    actual = [
        {
            "path": source.relative_to(root).as_posix(),
            "source_sha256": _file_sha256(source),
        }
        for source in sorted((root / "src").rglob("*.py"))
        if source.is_file() and not source.is_symlink()
    ]
    if observed != actual:
        raise PatchAcceptanceError("baseline src inventory drifted")


def _validate_hashed_inventory(value: object, root: Path, label: str) -> None:
    entries = object_list(value, f"baseline {label} inventory")
    if not entries:
        raise PatchAcceptanceError(f"baseline {label} inventory is empty")
    paths: list[str] = []
    for raw in entries:
        item = mapping(raw, f"baseline {label} entry")
        _exact_keys(item, {"path", "source_sha256"}, f"baseline {label} entry")
        relative = _safe_artifact_path(
            item.get("path"), f"baseline {label} path"
        )
        source = root / relative
        if source.is_symlink() or not source.is_file():
            raise PatchAcceptanceError(f"baseline {label} path is missing")
        if _file_sha256(source) != _sha256(
            item.get("source_sha256"), f"baseline {label} digest"
        ):
            raise PatchAcceptanceError(f"baseline {label} inventory drifted")
        paths.append(relative.as_posix())
    if paths != sorted(paths):
        raise PatchAcceptanceError(f"baseline {label} inventory is not sorted")
    _unique(paths, f"baseline {label} path")


def _validate_symbol_inventory_link(
    source_symbols_path: Path, source_inventory: object, root: Path
) -> None:
    payload = _load_mapping(source_symbols_path, "source symbols")
    symbols = object_list(payload.get("symbols"), "source symbols")
    source_entries = object_list(source_inventory, "baseline src inventory")
    hashes = {
        _safe_source_path(
            mapping(entry, "baseline src entry").get("path")
        ).as_posix(): _sha256(
            mapping(entry, "baseline src entry").get("source_sha256"),
            "baseline src digest",
        )
        for entry in source_entries
    }
    for raw in symbols:
        symbol = mapping(raw, "source symbol")
        relative = _safe_source_path(symbol.get("path")).as_posix()
        if hashes.get(relative) != _file_sha256(root / relative):
            raise PatchAcceptanceError(
                "source symbol manifest is not bound to baseline inventory"
            )


def _validate_baseline_evidence(
    path: Path, decisions_path: Path, root: Path
) -> None:
    payload = _load_mapping(path, "baseline evidence")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "phase",
            "patch_tools",
            "public_tool_inventory",
            "event_inventory",
            "migration_inventory",
            "server_route_inventory",
            "protocol_inventory",
            "context_capability_inventory",
            "section2_facts",
            "runtime_patch_advertisement",
            "contract_inventory_sha256",
            "evidence_sha256",
        },
        "baseline evidence",
    )
    _header(payload, "baseline evidence")
    if payload.get("phase") != _CURRENT_PHASE:
        raise PatchAcceptanceError("baseline evidence phase is invalid")
    if payload.get("patch_tools") != []:
        raise PatchAcceptanceError("Phase 0 cannot advertise patch tools")
    _validate_section2_facts(payload.get("section2_facts"), root)
    _validate_runtime_patch_advertisement(
        payload.get("runtime_patch_advertisement"), root
    )
    decisions = _load_mapping(decisions_path, "contract decisions")
    inventories = mapping(decisions.get("inventories"), "baseline inventories")
    if payload.get("contract_inventory_sha256") != decisions.get(
        "inventory_sha256"
    ):
        raise PatchAcceptanceError(
            "baseline evidence is not bound to contract inventory"
        )
    expected_inventory_fields = {
        "public_tool_inventory": "public_tools",
        "event_inventory": "events",
        "migration_inventory": "migrations",
        "server_route_inventory": "server_routes",
        "protocol_inventory": "protocols",
        "context_capability_inventory": "context_capabilities",
    }
    for field, inventory_field in expected_inventory_fields.items():
        values = object_list(payload.get(field), field)
        if not values:
            raise PatchAcceptanceError(f"baseline {field} is empty")
        if values != inventories[inventory_field]:
            raise PatchAcceptanceError(
                f"baseline {field} differs from contract inventory"
            )
    canonical = {
        key: value
        for key, value in payload.items()
        if key != "evidence_sha256"
    }
    if payload.get("evidence_sha256") != canonical_sha256(canonical):
        raise PatchAcceptanceError("baseline evidence digest is invalid")


def _validate_section2_facts(value: object, root: Path) -> None:
    """Verify the re-audited repository observations have live sources."""
    facts = object_list(value, "Section 2 starting-point facts")
    expected_ids = tuple(f"PATCH-S2-{index:03d}" for index in range(1, 12))
    observed_ids: list[str] = []
    markers: set[tuple[str, int]] = set()
    for raw in facts:
        fact = mapping(raw, "Section 2 starting-point fact")
        _exact_keys(
            fact,
            {"id", "observation", "source_markers"},
            "Section 2 starting-point fact",
        )
        observed_ids.append(
            _string(fact.get("id"), "Section 2 starting-point fact ID")
        )
        _string(
            fact.get("observation"), "Section 2 starting-point observation"
        )
        fact_markers = object_list(
            fact.get("source_markers"), "Section 2 source markers"
        )
        if not fact_markers:
            raise PatchAcceptanceError(
                "Section 2 starting-point fact has no source markers"
            )
        for raw_marker in fact_markers:
            marker = mapping(raw_marker, "Section 2 source marker")
            _exact_keys(
                marker,
                {"path", "line", "text", "source_sha256"},
                "Section 2 source marker",
            )
            relative = _safe_section2_source_path(marker.get("path"))
            line = _line(marker.get("line"), "Section 2 source line")
            text = _string(marker.get("text"), "Section 2 source text")
            digest = _sha256(
                marker.get("source_sha256"), "Section 2 source digest"
            )
            source = root / relative
            if source.is_symlink() or not source.is_file():
                raise PatchAcceptanceError(
                    "Section 2 source marker does not name a regular file"
                )
            lines = source.read_text(encoding="utf-8").splitlines()
            if line > len(lines) or lines[line - 1] != text:
                raise PatchAcceptanceError("Section 2 source marker drifted")
            if _file_sha256(source) != digest:
                raise PatchAcceptanceError("Section 2 source digest drifted")
            identity = (relative.as_posix(), line)
            if identity in markers:
                raise PatchAcceptanceError(
                    "Section 2 source marker is duplicated"
                )
            markers.add(identity)
    if tuple(observed_ids) != expected_ids:
        raise PatchAcceptanceError(
            "Section 2 starting-point facts are incomplete"
        )


def _safe_section2_source_path(value: object) -> Path:
    """Return one tracked implementation or gate source path."""
    relative = _safe_artifact_path(
        value, "Section 2 source path", suffix=".py"
    )
    if not relative.parts or relative.parts[0] not in {"scripts", "src"}:
        raise PatchAcceptanceError("Section 2 source path is invalid")
    return relative


def _validate_runtime_patch_advertisement(value: object, root: Path) -> None:
    """Prove Phase 0 has no runtime patch package or public selector."""
    payload = mapping(value, "runtime patch advertisement evidence")
    _exact_keys(
        payload,
        {"forbidden_paths", "forbidden_tokens", "runtime_probe"},
        "runtime patch advertisement evidence",
    )
    paths = tuple(
        _safe_artifact_path(raw, "forbidden patch path")
        for raw in object_list(
            payload.get("forbidden_paths"), "forbidden paths"
        )
    )
    if paths != (
        Path("src/avalan/patch"),
        Path("src/avalan/tool/patch.py"),
    ):
        raise PatchAcceptanceError("runtime patch path evidence is incomplete")
    for path in paths:
        if (root / path).exists():
            raise PatchAcceptanceError("Phase 0 runtime patch path is present")
    tokens = tuple(
        _string(raw, "forbidden patch advertisement token")
        for raw in object_list(
            payload.get("forbidden_tokens"), "forbidden patch tokens"
        )
    )
    expected_tokens = (
        "patch.edit",
        "patch.apply",
        'namespace="patch"',
        "namespace='patch'",
    )
    if tokens != expected_tokens:
        raise PatchAcceptanceError(
            "runtime patch token evidence is incomplete"
        )
    for source in (root / "src").rglob("*.py"):
        if source.is_symlink() or not source.is_file():
            raise PatchAcceptanceError("runtime source inventory is invalid")
        contents = source.read_text(encoding="utf-8")
        if any(token in contents for token in tokens):
            raise PatchAcceptanceError(
                "Phase 0 runtime patch advertisement is present"
            )
    observed = _runtime_patch_incapability_probe()
    expected = tuple(
        _string(item, "runtime patch probe surface")
        for item in object_list(
            payload.get("runtime_probe"), "runtime patch probe surfaces"
        )
    )
    if expected != observed:
        raise PatchAcceptanceError("runtime patch incapability probe drifted")


def _runtime_patch_incapability_probe() -> tuple[str, ...]:
    """Exercise real configured discovery surfaces before patch exists."""
    from fastapi import FastAPI
    from starlette.routing import Route

    from avalan.agent.loader import OrchestratorLoader
    from avalan.cli.__main__ import CLI
    from avalan.flow.registry import (
        default_flow_node_registry,
        tool_flow_node_registry,
    )
    from avalan.model.capability import ModelCapabilityCatalog
    from avalan.server.a2a.router import install_a2a_routes
    from avalan.server.authority import REMOTE_CONTAINER_PROFILE_SELECTOR_KEYS
    from avalan.server.routers.mcp import create_router
    from avalan.task.definition import TaskTargetType
    from avalan.task.target import (
        CallableTaskTargetRunner,
        TaskTargetRunnerRegistry,
    )
    from avalan.tool import ToolSet
    from avalan.tool.manager import ToolManager

    default_manager = ToolManager.create_instance()
    _assert_runtime_patch_identity_absent(
        descriptor.name for descriptor in default_manager.list_tools()
    )
    if (
        default_manager.resolve_tool_name("patch.edit").canonical_name
        is not None
    ):
        raise PatchAcceptanceError("default ToolManager advertises patch")
    configured_toolsets = (
        ToolSet(namespace="phase0", tools=(_runtime_probe_tool,)),
    )
    _assert_runtime_toolsets_incapable(configured_toolsets)
    manager = ToolManager.create_instance(
        available_toolsets=configured_toolsets
    )
    _assert_runtime_patch_identity_absent(
        descriptor.name for descriptor in manager.list_tools()
    )
    if (
        not manager.list_tools()
        or manager.resolve_tool_name("patch.edit").canonical_name is not None
    ):
        raise PatchAcceptanceError("ToolManager advertises patch")
    seed = manager.export_model_capability_seed()
    catalog = ModelCapabilityCatalog.create(seed)
    _assert_runtime_patch_identity_absent(
        descriptor.canonical_name
        for descriptor in catalog.domain_seed.descriptors
    )
    registry = default_flow_node_registry()
    if registry.supports("patch") or registry.supports("patch.edit"):
        raise PatchAcceptanceError("flow registry advertises patch")
    configured_flow = tool_flow_node_registry(manager, base_registry=registry)
    if configured_flow.supports("patch") or configured_flow.supports(
        "patch.edit"
    ):
        raise PatchAcceptanceError("configured flow registry advertises patch")
    task_registry = TaskTargetRunnerRegistry(
        default=CallableTaskTargetRunner(_runtime_task_target_probe)
    )
    _assert_runtime_patch_identity_absent(
        target.value for target in TaskTargetType
    )
    if any(
        task_registry.supports_durable_resume(target)
        for target in TaskTargetType
    ):
        raise PatchAcceptanceError("task registry advertises patch durability")
    _assert_runtime_patch_identity_absent(
        candidate.__name__ for candidate in OrchestratorLoader.__subclasses__()
    )
    cli = CLI(getLogger("avalan.patch-phase0"))
    command_actions = tuple(
        action
        for action in cli._parser._actions
        if getattr(action, "dest", None) == "command"
    )
    choices = getattr(command_actions[0], "choices", None)
    if (
        len(command_actions) != 1
        or not isinstance(choices, dict)
        or "patch" in choices
    ):
        raise PatchAcceptanceError("CLI selector advertises patch")
    app = FastAPI()
    app.include_router(create_router(), prefix="/mcp")
    install_a2a_routes(app, prefix="/a2a", name="run", description=None)
    route_paths = tuple(
        route.path for route in app.routes if isinstance(route, Route)
    )
    _assert_runtime_patch_identity_absent(route_paths)
    if any("patch" in path for path in route_paths):
        raise PatchAcceptanceError("MCP server routes advertise patch")
    _assert_runtime_patch_identity_absent(app.openapi().get("paths", {}))
    _assert_runtime_patch_identity_absent(
        REMOTE_CONTAINER_PROFILE_SELECTOR_KEYS
    )
    return (
        "toolmanager_default_and_configured_discovery",
        "cli_selectors_and_commands",
        "mcp_a2a_server_openapi_routes",
        "flow_task_orchestrator_nodes",
        "target_handshake_profile_selectors",
        "provider_capability_catalog",
    )


def _runtime_probe_tool(value: str) -> str:
    """Return one configured non-patch tool value for runtime discovery."""
    return value


async def _runtime_task_target_probe(context: object) -> object:
    """Return inert task output while exercising the real task registry."""
    del context
    return None


def _assert_runtime_toolsets_incapable(toolsets: tuple[object, ...]) -> None:
    """Reject an injected patch namespace before ToolManager advertisement."""
    from avalan.tool import ToolSet

    for toolset in toolsets:
        if not isinstance(toolset, ToolSet):
            raise PatchAcceptanceError("runtime ToolSet probe is malformed")
        namespace = toolset.namespace
        _assert_runtime_patch_identity_absent(
            () if namespace is None else (namespace, f"{namespace}.probe")
        )


def _assert_runtime_patch_identity_absent(identities: Iterable[str]) -> None:
    """Reject direct or dynamically composed patch identities."""
    for identity in identities:
        lowered = identity.lower()
        if lowered == "patch" or lowered.startswith("patch."):
            raise PatchAcceptanceError(
                "dynamic runtime advertisement is present"
            )


def _validate_phase_evidence(
    path: Path, manifest: AcceptanceManifest, root: Path
) -> None:
    """Validate the complete, immutable evidence record for Phase 0."""
    payload = _load_mapping(path, "phase evidence")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "phase",
            "status",
            "scope",
            "recorded_on",
            "ownership",
            "changed_paths",
            "exact_gate",
            "commands",
            "artifact_digests",
            "suite_facts",
            "coverage",
            "node_counts",
            "quality_gates",
            "database_lifecycle",
            "profiles",
            "e2e_ids",
            "redaction_and_bounds",
            "review",
            "active_node_ids",
            "external_effects",
            "record_sha256",
        },
        "phase evidence",
    )
    _header(payload, "phase evidence")
    if payload.get("phase") != _CURRENT_PHASE:
        raise PatchAcceptanceError("phase evidence phase is invalid")
    status = _string(payload.get("status"), "phase evidence status")
    if status not in {
        "in_progress",
        "complete",
    }:
        raise PatchAcceptanceError("phase evidence status is invalid")
    _string(payload.get("scope"), "phase evidence scope")
    if (
        _string(payload.get("recorded_on"), "phase evidence date")
        != "2026-08-08"
    ):
        raise PatchAcceptanceError("phase evidence date is invalid")
    _validate_phase_evidence_ownership(payload.get("ownership"))
    _validate_phase_evidence_changed_paths(payload.get("changed_paths"))
    exact_gate_complete = _validate_phase_evidence_exact_gate(
        payload.get("exact_gate"), status
    )
    command_ids = _validate_phase_evidence_commands(
        payload.get("commands"), exact_gate_complete
    )
    _validate_phase_evidence_artifacts(
        payload.get("artifact_digests"), root, exact_gate_complete
    )
    active = tuple(
        _test_node(value)
        for value in object_list(
            payload.get("active_node_ids"), "phase evidence nodes"
        )
    )
    if set(active) != {
        node.node_id for node in manifest.active_nodes(_CURRENT_PHASE)
    }:
        raise PatchAcceptanceError(
            "phase evidence active nodes differ from manifest"
        )
    _validate_phase_evidence_suites(
        payload.get("suite_facts"), active, command_ids, exact_gate_complete
    )
    _validate_phase_evidence_coverage(
        payload.get("coverage"), root, exact_gate_complete
    )
    _validate_phase_evidence_counts(payload.get("node_counts"), manifest)
    quality_complete = _validate_phase_evidence_quality(
        payload.get("quality_gates"), command_ids, exact_gate_complete
    )
    _validate_phase_evidence_database(payload.get("database_lifecycle"))
    _validate_phase_evidence_profiles(
        payload.get("profiles"), payload.get("e2e_ids"), command_ids
    )
    redaction_complete = _validate_phase_evidence_redaction(
        payload.get("redaction_and_bounds")
    )
    review_open_blocker = _validate_phase_evidence_review(
        payload.get("review")
    )
    effects = mapping(
        payload.get("external_effects"), "phase external effects"
    )
    _exact_keys(
        effects,
        {
            "deployment",
            "publication",
            "live_call",
            "workspace_mutation",
            "production_activation",
        },
        "phase external effects",
    )
    if any(value is not False for value in effects.values()):
        raise PatchAcceptanceError(
            "Phase 0 evidence claims an unauthorized external effect"
        )
    if status == "complete":
        if not exact_gate_complete:
            raise PatchAcceptanceError(
                "complete phase evidence requires retained exact-gate "
                "artifacts"
            )
        if review_open_blocker:
            raise PatchAcceptanceError(
                "complete phase evidence has unresolved P0-P2 review findings"
            )
        if not quality_complete or not redaction_complete:
            raise PatchAcceptanceError(
                "complete phase evidence has incomplete quality gates"
            )
    canonical = {
        key: value for key, value in payload.items() if key != "record_sha256"
    }
    if payload.get("record_sha256") != canonical_sha256(canonical):
        raise PatchAcceptanceError("phase evidence digest is invalid")


def _validate_phase_evidence_exact_gate(value: object, status: str) -> bool:
    """Require an explicit pending or retained exact-gate state."""
    gate = mapping(value, "phase evidence exact gate")
    _exact_keys(gate, {"status", "reason"}, "phase evidence exact gate")
    gate_status = _string(
        gate.get("status"), "phase evidence exact gate status"
    )
    _string(gate.get("reason"), "phase evidence exact gate reason")
    if status == "in_progress":
        if gate_status != "pending":
            raise PatchAcceptanceError(
                "in-progress evidence must keep exact gate pending"
            )
        return False
    if gate_status != "complete":
        raise PatchAcceptanceError(
            "complete evidence exact gate is not complete"
        )
    return True


def _validate_phase_evidence_ownership(value: object) -> None:
    """Validate named owners and immutable base/head provenance."""
    ownership = mapping(value, "phase evidence ownership")
    _exact_keys(
        ownership,
        {"implementation_owners", "review_owners", "base_sha", "head_sha"},
        "phase evidence ownership",
    )
    for field in ("implementation_owners", "review_owners"):
        owners = tuple(
            _string(item, f"phase evidence {field}")
            for item in object_list(ownership.get(field), field)
        )
        if not owners or len(set(owners)) != len(owners):
            raise PatchAcceptanceError("phase evidence owners are invalid")
    for field in ("base_sha", "head_sha"):
        sha = _string(ownership.get(field), f"phase evidence {field}")
        if compile_regex(r"^[0-9a-f]{40}$").fullmatch(sha) is None:
            raise PatchAcceptanceError(
                "phase evidence git provenance is invalid"
            )


def _validate_phase_evidence_changed_paths(value: object) -> None:
    """Require one sorted, bounded, non-directory changed-path inventory."""
    paths = tuple(
        _string(item, "phase evidence changed path")
        for item in object_list(value, "phase evidence changed paths")
    )
    if (
        not paths
        or paths != tuple(sorted(paths))
        or len(set(paths)) != len(paths)
    ):
        raise PatchAcceptanceError("phase evidence changed paths are invalid")
    for raw in paths:
        path = PurePosixPath(raw)
        if path.is_absolute() or ".." in path.parts or raw.endswith("/"):
            raise PatchAcceptanceError("phase evidence changed path is unsafe")


def _validate_phase_evidence_commands(
    value: object, exact_gate_complete: bool
) -> frozenset[str]:
    """Validate exact command receipts with duration and safe environment."""
    identifiers: set[str] = set()
    for raw in object_list(value, "phase evidence commands"):
        command = mapping(raw, "phase evidence command")
        expected = {"id", "command", "status", "reason"}
        if exact_gate_complete:
            expected |= {
                "exit_code",
                "duration_ms",
                "environment",
                "output_sha256",
            }
        _exact_keys(command, expected, "phase evidence command")
        identifier = _identifier(
            command.get("id"), "phase evidence command ID"
        )
        identifiers.add(identifier)
        _string(command.get("command"), "phase evidence command")
        command_status = _string(
            command.get("status"), "phase evidence command status"
        )
        _string(command.get("reason"), "phase evidence command reason")
        if not exact_gate_complete:
            if command_status != "pending":
                raise PatchAcceptanceError(
                    "in-progress phase evidence cannot claim a passed command"
                )
            continue
        if command_status != "passed":
            raise PatchAcceptanceError(
                "complete phase evidence command is pending"
            )
        if (
            _nonnegative_int(command.get("exit_code"), "phase command exit")
            != 0
        ):
            raise PatchAcceptanceError("phase evidence command did not pass")
        if (
            _nonnegative_int(
                command.get("duration_ms"), "phase command duration"
            )
            == 0
        ):
            raise PatchAcceptanceError(
                "phase evidence command duration is invalid"
            )
        _sha256(command.get("output_sha256"), "phase command output digest")
        environment = mapping(
            command.get("environment"), "phase command environment"
        )
        _exact_keys(
            environment,
            {"python", "profile", "configuration"},
            "phase command environment",
        )
        _string(environment.get("python"), "phase command Python")
        _string(environment.get("profile"), "phase command profile")
        configuration = mapping(
            environment.get("configuration"), "phase command configuration"
        )
        for name, raw_setting in configuration.items():
            setting = _string(name, "phase command configuration name")
            _string(raw_setting, "phase command configuration value")
            if any(
                token in setting.lower()
                for token in ("secret", "token", "password")
            ):
                raise PatchAcceptanceError(
                    "phase evidence configuration exposes secret naming"
                )
    if not identifiers:
        raise PatchAcceptanceError("phase evidence has no command receipts")
    if len(identifiers) != len(object_list(value, "phase evidence commands")):
        raise PatchAcceptanceError("phase evidence command IDs are duplicated")
    return frozenset(identifiers)


def _validate_phase_evidence_artifacts(
    value: object, root: Path, exact_gate_complete: bool
) -> None:
    """Require a sorted immutable digest inventory for Phase 0 artifacts."""
    raw_artifacts = object_list(value, "phase evidence artifact digests")
    if not exact_gate_complete:
        if raw_artifacts:
            raise PatchAcceptanceError(
                "in-progress phase evidence cannot retain stale artifacts"
            )
        return
    paths: list[str] = []
    names: set[str] = set()
    for raw in raw_artifacts:
        artifact = mapping(raw, "phase evidence artifact digest")
        _exact_keys(
            artifact,
            {"name", "path", "sha256"},
            "phase evidence artifact digest",
        )
        names.add(_string(artifact.get("name"), "phase artifact name"))
        path = _string(artifact.get("path"), "phase artifact path")
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise PatchAcceptanceError("phase artifact path is unsafe")
        paths.append(path)
        expected_sha = _sha256(artifact.get("sha256"), "phase artifact digest")
        source = root / candidate
        if source.is_symlink() or not source.is_file():
            raise PatchAcceptanceError("phase artifact path is missing")
        if _file_sha256(source) != expected_sha:
            raise PatchAcceptanceError("phase artifact digest drifted")
    if (
        len(names) != len(paths)
        or paths != sorted(paths)
        or len(set(paths)) != len(paths)
    ):
        raise PatchAcceptanceError(
            "phase artifact digest inventory is invalid"
        )


def _validate_phase_evidence_suites(
    value: object,
    active_nodes: tuple[str, ...],
    command_ids: frozenset[str],
    exact_gate_complete: bool,
) -> None:
    """Validate collected-node and result facts against active nodes."""
    facts = object_list(value, "phase evidence suite facts")
    if not exact_gate_complete:
        if facts:
            raise PatchAcceptanceError(
                "in-progress phase evidence cannot claim suite success"
            )
        return
    node_ids: set[str] = set()
    for raw in facts:
        suite = mapping(raw, "phase evidence suite fact")
        _exact_keys(
            suite,
            {
                "node_id",
                "command_id",
                "outcome",
                "collected_nodes",
                "passed_nodes",
                "skipped_nodes",
                "xfail_nodes",
                "xpass_nodes",
                "deselected_nodes",
                "placeholder_nodes",
                "ignored_active_nodes",
            },
            "phase evidence suite fact",
        )
        node = _test_node(suite.get("node_id"))
        node_ids.add(node)
        if (
            _identifier(suite.get("command_id"), "phase suite command")
            not in command_ids
        ):
            raise PatchAcceptanceError(
                "phase suite references unknown command"
            )
        if _string(suite.get("outcome"), "phase suite outcome") != "passed":
            raise PatchAcceptanceError("phase suite did not pass")
        collected = _nonnegative_int(
            suite.get("collected_nodes"), "phase suite collected nodes"
        )
        passed = _nonnegative_int(
            suite.get("passed_nodes"), "phase suite passed nodes"
        )
        if not collected or passed != collected:
            raise PatchAcceptanceError("phase suite result counts are invalid")
        for field in (
            "skipped_nodes",
            "xfail_nodes",
            "xpass_nodes",
            "deselected_nodes",
            "placeholder_nodes",
            "ignored_active_nodes",
        ):
            if _nonnegative_int(suite.get(field), f"phase suite {field}") != 0:
                raise PatchAcceptanceError(
                    "phase suite contains non-executable evidence"
                )
    if node_ids != set(active_nodes):
        raise PatchAcceptanceError(
            "phase suite node facts differ from active nodes"
        )


def _validate_phase_evidence_coverage(
    value: object, root: Path, exact_gate_complete: bool
) -> None:
    """Require the exact fresh coverage totals and immutable report digests."""
    coverage = mapping(value, "phase evidence coverage")
    if not exact_gate_complete:
        _exact_keys(
            coverage,
            {"status", "reason"},
            "phase evidence coverage",
        )
        if (
            _string(coverage.get("status"), "phase coverage status")
            != "pending"
        ):
            raise PatchAcceptanceError("in-progress coverage must be pending")
        _string(coverage.get("reason"), "phase coverage pending reason")
        return
    _exact_keys(
        coverage,
        {
            "tool",
            "collected_pytest_nodes",
            "nonempty_src_files",
            "statements",
            "covered_statements",
            "missing_statements",
            "report_sha256",
            "per_file_sha256",
        },
        "phase evidence coverage",
    )
    if _string(coverage.get("tool"), "phase coverage tool") != "coverage.py":
        raise PatchAcceptanceError("phase coverage tool is invalid")
    reports = _phase_evidence_artifact_paths(root)
    try:
        verification = verify_src_coverage(reports[0], repo_root=root)
    except CoverageVerificationError as exc:
        raise PatchAcceptanceError(str(exc)) from exc
    raw_report = mapping(strict_json_path(reports[0]), "coverage report")
    files = mapping(raw_report.get("files"), "coverage files")
    expected = {
        "collected_pytest_nodes": _pytest_facts_collected(reports[2]),
        "nonempty_src_files": len(verification.files),
        "statements": verification.summary.num_statements,
        "covered_statements": verification.summary.covered_lines,
        "missing_statements": verification.summary.missing_lines,
    }
    for field, expected_value in expected.items():
        if (
            _nonnegative_int(coverage.get(field), f"phase coverage {field}")
            != expected_value
        ):
            raise PatchAcceptanceError("phase coverage totals are stale")
    if _sha256(
        coverage.get("report_sha256"), "phase coverage report digest"
    ) != _file_sha256(reports[0]):
        raise PatchAcceptanceError("phase coverage report digest is stale")
    if _sha256(
        coverage.get("per_file_sha256"), "phase coverage per-file digest"
    ) != canonical_sha256(files):
        raise PatchAcceptanceError("phase coverage per-file digest is stale")
    if not reports[1].is_file() or reports[1].is_symlink():
        raise PatchAcceptanceError("phase coverage XML artifact is missing")


def _phase_evidence_artifact_paths(root: Path) -> tuple[Path, Path, Path]:
    """Return retained gate artifacts, preferring explicit mirror paths."""
    defaults = (
        root / "coverage.json",
        root / "coverage.xml",
        root / ".patch-contract-pytest-facts.json",
    )
    values = tuple(environ.get(name) for name in _PHASE_EVIDENCE_ARTIFACT_ENVS)
    if any(value is None for value in values) and any(
        value is not None for value in values
    ):
        raise PatchAcceptanceError(
            "phase evidence artifact environment is partial"
        )
    paths = (
        defaults
        if not any(values)
        else tuple(
            Path(value).resolve() for value in values if value is not None
        )
    )
    if len(paths) != 3 or any(path.parent != root.resolve() for path in paths):
        raise PatchAcceptanceError(
            "phase evidence artifact path escapes execution root"
        )
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise PatchAcceptanceError(
                "phase evidence retained artifact is missing"
            )
    first, second, third = paths
    return first, second, third


def _pytest_facts_collected(path: Path) -> int:
    """Return the collected count from the gate-owned pytest receipt."""
    facts = mapping(strict_json_path(path), "phase pytest facts")
    return _nonnegative_int(facts.get("collected"), "phase pytest collected")


def _validate_phase_evidence_counts(
    value: object, manifest: AcceptanceManifest
) -> None:
    """Validate active/planned requirement and acceptance inventory counts."""
    counts = mapping(value, "phase evidence node counts")
    _exact_keys(
        counts,
        {
            "active_requirements",
            "planned_requirements",
            "active_acceptance_nodes",
            "planned_acceptance_nodes",
        },
        "phase evidence node counts",
    )
    expected = {
        "active_requirements": 87,
        "planned_requirements": 930,
        "active_acceptance_nodes": len(manifest.active_nodes(_CURRENT_PHASE)),
        "planned_acceptance_nodes": (
            len(manifest.nodes) - len(manifest.active_nodes(_CURRENT_PHASE))
        ),
    }
    for field, expected_value in expected.items():
        if (
            _nonnegative_int(counts.get(field), f"phase evidence {field}")
            != expected_value
        ):
            raise PatchAcceptanceError(
                "phase evidence inventory counts are invalid"
            )


def _validate_phase_evidence_quality(
    value: object,
    command_ids: frozenset[str],
    exact_gate_complete: bool,
) -> bool:
    """Require named Phase 0 quality receipts with a passing command owner."""
    quality = mapping(value, "phase evidence quality gates")
    expected = {
        "project_mypy",
        "patch_mypy",
        "positive_type_assertions",
        "rejected_type_diagnostics",
        "blocking_io_scan",
        "async_heartbeat",
        "resource_leaks",
    }
    _exact_keys(quality, expected, "phase evidence quality gates")
    complete = True
    for name in expected:
        item = mapping(quality.get(name), f"phase quality {name}")
        _exact_keys(
            item, {"status", "command_id", "evidence"}, f"phase quality {name}"
        )
        status = _string(item.get("status"), f"phase quality {name} status")
        expected_status = "passed" if exact_gate_complete else "pending"
        if status != expected_status:
            raise PatchAcceptanceError("phase quality status is invalid")
        complete = complete and status == "passed"
        if (
            _identifier(
                item.get("command_id"), f"phase quality {name} command"
            )
            not in command_ids
        ):
            raise PatchAcceptanceError(
                "phase quality references unknown command"
            )
        _string(item.get("evidence"), f"phase quality {name} evidence")
    return complete


def _validate_phase_evidence_database(value: object) -> None:
    """Record the explicit non-applicable Phase 0 database lifecycle facts."""
    database = mapping(value, "phase evidence database lifecycle")
    _exact_keys(
        database,
        {
            "migration_revision",
            "schema_identity",
            "crash_restart_boundary",
            "encryption_key_fixture_version",
            "outbox_settlement",
            "teardown_proof",
        },
        "phase evidence database lifecycle",
    )
    for field in database:
        if (
            _string(database.get(field), f"phase database {field}")
            != "not_applicable_phase_0"
        ):
            raise PatchAcceptanceError("phase database lifecycle is invalid")


def _validate_phase_evidence_profiles(
    value: object, e2e_value: object, command_ids: frozenset[str]
) -> None:
    """Validate real/scripted profile declarations and public E2E inventory."""
    e2e_ids = tuple(
        _identifier(item, "phase E2E ID")
        for item in object_list(e2e_value, "phase E2E IDs")
    )
    if len(set(e2e_ids)) != len(e2e_ids):
        raise PatchAcceptanceError("phase E2E IDs are duplicated")
    profile_ids: set[str] = set()
    for raw in object_list(value, "phase evidence profiles"):
        profile = mapping(raw, "phase evidence profile")
        _exact_keys(
            profile,
            {"id", "kind", "contexts", "command_id", "e2e_ids", "receipt"},
            "phase evidence profile",
        )
        profile_ids.add(_identifier(profile.get("id"), "phase profile ID"))
        if _string(profile.get("kind"), "phase profile kind") not in {
            "scripted",
            "real",
            "inactive",
        }:
            raise PatchAcceptanceError("phase profile kind is invalid")
        contexts = tuple(
            _string(item, "phase profile context")
            for item in object_list(
                profile.get("contexts"), "phase profile contexts"
            )
        )
        if not contexts or len(set(contexts)) != len(contexts):
            raise PatchAcceptanceError("phase profile contexts are invalid")
        if (
            _identifier(profile.get("command_id"), "phase profile command")
            not in command_ids
        ):
            raise PatchAcceptanceError(
                "phase profile references unknown command"
            )
        profile_e2e = tuple(
            _identifier(item, "phase profile E2E")
            for item in object_list(
                profile.get("e2e_ids"), "phase profile E2E IDs"
            )
        )
        if not set(profile_e2e).issubset(set(e2e_ids)):
            raise PatchAcceptanceError("phase profile references unknown E2E")
        _string(profile.get("receipt"), "phase profile receipt")
    if not profile_ids or len(profile_ids) != len(
        object_list(value, "phase evidence profiles")
    ):
        raise PatchAcceptanceError("phase profile IDs are invalid")


def _validate_phase_evidence_redaction(value: object) -> bool:
    """Validate redaction, disclosure, bounds, and diagnostic facts."""
    redaction = mapping(value, "phase evidence redaction and bounds")
    _exact_keys(
        redaction,
        {
            "redaction_canary",
            "renderer_injection",
            "timing_disclosure",
            "fuzz",
            "liveness",
            "retention",
            "diagnostic_separation",
            "max_request_bytes",
            "max_duration_ticks",
        },
        "phase evidence redaction and bounds",
    )
    complete = True
    for field in (
        "redaction_canary",
        "renderer_injection",
        "timing_disclosure",
        "fuzz",
        "liveness",
        "retention",
        "diagnostic_separation",
    ):
        status = _string(redaction.get(field), f"phase redaction {field}")
        if status not in {
            "passed",
            "not_run",
            "not_applicable",
        }:
            raise PatchAcceptanceError("phase redaction status is invalid")
        complete = complete and status != "not_run"
    for field in ("max_request_bytes", "max_duration_ticks"):
        if (
            _nonnegative_int(redaction.get(field), f"phase redaction {field}")
            == 0
        ):
            raise PatchAcceptanceError("phase redaction bound is invalid")
    return complete


def _validate_phase_evidence_review(value: object) -> bool:
    """Return whether independent review still has a blocking open finding."""
    review = mapping(value, "phase evidence review")
    _exact_keys(review, {"rounds"}, "phase evidence review")
    rounds = object_list(review.get("rounds"), "phase review rounds")
    if not rounds:
        raise PatchAcceptanceError("phase evidence review rounds are empty")
    blocked = False
    finding_ids: set[str] = set()
    previous_round = 0
    finding_count = 0
    for raw in rounds:
        round_record = mapping(raw, "phase review round")
        _exact_keys(
            round_record,
            {"round", "findings", "residual_risks"},
            "phase review round",
        )
        number = round_record.get("round")
        if type(number) is not int or not 1 <= number <= 5:
            raise PatchAcceptanceError(
                "phase evidence review round is invalid"
            )
        if number <= previous_round:
            raise PatchAcceptanceError(
                "phase evidence review rounds are unordered"
            )
        previous_round = number
        findings = object_list(
            round_record.get("findings"), "phase review findings"
        )
        for finding_raw in findings:
            finding = mapping(finding_raw, "phase review finding")
            _exact_keys(
                finding,
                {
                    "id",
                    "severity",
                    "disposition",
                    "owner",
                    "evidence",
                    "rationale",
                },
                "phase review finding",
            )
            identifier = _identifier(
                finding.get("id"), "phase review finding ID"
            )
            if identifier in finding_ids:
                raise PatchAcceptanceError(
                    "phase review finding IDs are invalid"
                )
            finding_ids.add(identifier)
            finding_count += 1
            severity = _string(
                finding.get("severity"), "phase review severity"
            )
            disposition = _string(
                finding.get("disposition"), "phase review disposition"
            )
            if severity not in {"P0", "P1", "P2", "P3", "P4"} or (
                disposition not in {"fixed", "accepted", "open"}
            ):
                raise PatchAcceptanceError("phase review finding is invalid")
            for field in ("owner", "evidence", "rationale"):
                _string(finding.get(field), f"phase review {field}")
            blocked = blocked or (
                severity in {"P0", "P1", "P2"} and disposition == "open"
            )
        for risk in object_list(
            round_record.get("residual_risks"), "phase residual risks"
        ):
            _string(risk, "phase residual risk")
    if not finding_ids or len(finding_ids) != finding_count:
        raise PatchAcceptanceError("phase review finding IDs are invalid")
    return blocked


def _require_fixture_bundle(directory: Path) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise PatchAcceptanceError("patch fixture directory is missing")
    for name in _FIXTURE_NAMES:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise PatchAcceptanceError(
                f"required patch fixture is missing: {name}"
            )


def _python_paths(root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for prefix in ("src", "scripts", "tests"):
        directory = root / prefix
        if not directory.is_dir():
            continue
        paths.extend(
            path
            for path in directory.rglob("*.py")
            if path.is_file() and not path.is_symlink()
        )
    return tuple(sorted(paths))


def _audit_source_artifact_open(
    event: str,
    arguments: tuple[object, ...],
) -> None:
    """Reject guarded interpreter-level opens of one forbidden design path."""
    if event != "open" or not _SOURCE_ARTIFACT_AUDIT_ENABLED.get():
        return
    if not arguments:
        return
    candidate = arguments[0]
    if not isinstance(candidate, (Path, bytes, str)):
        return
    if isinstance(candidate, bytes):
        try:
            value = candidate.decode("utf-8")
        except UnicodeDecodeError:
            return
    else:
        value = str(candidate)
    if _is_forbidden_source_artifact(value):
        raise PatchAcceptanceError(
            "tracked source opens ignored patch design artifact at runtime"
        )


addaudithook(_audit_source_artifact_open)


@dataclass(frozen=True, kw_only=True, slots=True)
class ArtifactAliases:
    """Store the imported names that can construct or read one path."""

    path_constructors: frozenset[str]
    path_modules: frozenset[str]
    os_modules: frozenset[str]
    io_modules: frozenset[str]
    builtin_modules: frozenset[str]
    reader_names: frozenset[str]
    reader_aliases: frozenset[str]
    getattr_names: frozenset[str]


def _artifact_aliases(tree: AST) -> ArtifactAliases:
    """Collect direct and aliased constructors and readers from one module."""
    path_constructors: set[str] = set()
    path_modules: set[str] = set()
    os_modules: set[str] = set()
    io_modules: set[str] = set()
    builtin_modules: set[str] = set()
    reader_names = {"open", "strict_json_path"}
    reader_aliases: set[str] = set()
    getattr_names = {"getattr"}
    for node in walk(tree):
        if isinstance(node, Import):
            for imported in node.names:
                _record_module_alias(
                    imported,
                    path_modules,
                    os_modules,
                    io_modules,
                    builtin_modules,
                )
        elif isinstance(node, ImportFrom):
            _record_from_alias(
                node,
                path_constructors,
                reader_aliases,
                getattr_names,
            )
    return ArtifactAliases(
        path_constructors=frozenset(path_constructors),
        path_modules=frozenset(path_modules),
        os_modules=frozenset(os_modules),
        io_modules=frozenset(io_modules),
        builtin_modules=frozenset(builtin_modules),
        reader_names=frozenset(reader_names),
        reader_aliases=frozenset(reader_aliases),
        getattr_names=frozenset(getattr_names),
    )


def _record_module_alias(
    imported: alias,
    path_modules: set[str],
    os_modules: set[str],
    io_modules: set[str],
    builtin_modules: set[str],
) -> None:
    """Record a module import whose attributes can address a path."""
    name = imported.asname or imported.name
    if imported.name == "pathlib":
        path_modules.add(name)
    elif imported.name == "os":
        os_modules.add(name)
    elif imported.name == "io":
        io_modules.add(name)
    elif imported.name == "builtins":
        builtin_modules.add(name)


def _record_from_alias(
    node: ImportFrom,
    path_constructors: set[str],
    reader_aliases: set[str],
    getattr_names: set[str],
) -> None:
    """Record one imported class or reader alias without trust by spelling."""
    module = node.module
    for imported in node.names:
        name = imported.asname or imported.name
        if module == "pathlib" and imported.name in {
            "Path",
            "PurePath",
            "PurePosixPath",
            "PureWindowsPath",
        }:
            path_constructors.add(name)
        if (
            module in {"builtins", "io", "os"} and imported.name == "open"
        ) or (
            module == "contract_gate" and imported.name == "strict_json_path"
        ):
            reader_aliases.add(name)
        if module == "builtins" and imported.name == "getattr":
            getattr_names.add(name)


def _artifact_path_values(
    tree: AST, aliases: ArtifactAliases
) -> dict[str, str]:
    """Resolve constant path variables before evaluating reader call sites."""
    values: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for node in walk(tree):
            target: Name | None = None
            value: expr | None = None
            if isinstance(node, Assign) and len(node.targets) == 1:
                candidate = node.targets[0]
                if isinstance(candidate, Name):
                    target = candidate
                    value = node.value
            elif isinstance(node, AnnAssign) and isinstance(node.target, Name):
                target = node.target
                value = node.value
            if target is None or value is None or target.id in values:
                continue
            resolved = _constant_path(value, values, aliases)
            if resolved is not None:
                values[target.id] = resolved
                changed = True
    return values


def _constant_path(
    value: expr, values: dict[str, str], aliases: ArtifactAliases
) -> str | None:
    """Resolve one literal path expression through safe finite constructors."""
    if isinstance(value, Constant) and isinstance(value.value, str):
        return value.value
    if isinstance(value, Name):
        return values.get(value.id)
    if isinstance(value, JoinedStr):
        fragments: list[str] = []
        for item in value.values:
            if isinstance(item, Constant) and isinstance(item.value, str):
                fragments.append(item.value)
                continue
            if isinstance(item, FormattedValue):
                resolved = _constant_path(item.value, values, aliases)
                if resolved is not None:
                    fragments.append(resolved)
                    continue
            return None
        return "".join(fragments)
    if isinstance(value, BinOp) and isinstance(value.op, (Add, Div)):
        left = _constant_path(value.left, values, aliases)
        right = _constant_path(value.right, values, aliases)
        if left is not None and right is not None:
            return _join_path_parts((left, right))
        return None
    if not isinstance(value, Call):
        return None
    if (
        isinstance(value.func, Attribute)
        and value.func.attr == "join"
        and isinstance(value.func.value, Constant)
        and isinstance(value.func.value.value, str)
        and len(value.args) == 1
    ):
        joined = _constant_path_sequence(value.args[0], values, aliases)
        if joined is not None:
            return value.func.value.value.join(joined)
    arguments = tuple(
        resolved
        for argument in value.args
        if (resolved := _constant_path(argument, values, aliases)) is not None
    )
    if len(arguments) != len(value.args):
        return None
    if _is_path_constructor(value.func, aliases):
        return _join_path_parts(arguments)
    if isinstance(value.func, Attribute) and value.func.attr in {
        "joinpath",
        "join",
    }:
        base = _constant_path(value.func.value, values, aliases)
        if base is not None:
            return _join_path_parts((base, *arguments))
    if _is_os_path_join(value.func, aliases):
        return _join_path_parts(arguments)
    return None


def _constant_path_sequence(
    value: expr,
    values: dict[str, str],
    aliases: ArtifactAliases,
) -> tuple[str, ...] | None:
    """Resolve one finite literal sequence used by ``str.join``."""
    if not isinstance(value, (List, Tuple)):
        return None
    resolved = tuple(
        _constant_path(item, values, aliases) for item in value.elts
    )
    if any(item is None for item in resolved):
        return None
    return tuple(item for item in resolved if item is not None)


def _join_path_parts(parts: tuple[str, ...]) -> str:
    """Return a normalized slash-separated path without filesystem IO."""
    return "/".join(part.strip("/") for part in parts if part != "")


def _is_path_constructor(value: expr, aliases: ArtifactAliases) -> bool:
    """Return whether one expression names an imported pathlib constructor."""
    if isinstance(value, Name):
        return value.id in aliases.path_constructors
    return (
        isinstance(value, Attribute)
        and isinstance(value.value, Name)
        and value.value.id in aliases.path_modules
        and value.attr
        in {"Path", "PurePath", "PurePosixPath", "PureWindowsPath"}
    )


def _is_os_path_join(value: expr, aliases: ArtifactAliases) -> bool:
    """Return whether one expression is an aliased ``os.path.join`` call."""
    return (
        isinstance(value, Attribute)
        and value.attr == "join"
        and isinstance(value.value, Attribute)
        and value.value.attr == "path"
        and isinstance(value.value.value, Name)
        and value.value.value.id in aliases.os_modules
    )


def _artifact_reader_aliases(
    tree: AST,
    aliases: ArtifactAliases,
    values: dict[str, str],
) -> frozenset[str]:
    """Resolve assignments that preserve a trusted reader alias."""
    readers = set(aliases.reader_aliases)
    changed = True
    while changed:
        changed = False
        for node in walk(tree):
            if not isinstance(node, Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, Name) or target.id in readers:
                continue
            if _is_direct_reader(node.value, aliases, readers, values):
                readers.add(target.id)
                changed = True
    return frozenset(readers)


def _read_artifact(
    call: Call,
    aliases: ArtifactAliases,
    values: dict[str, str],
    reader_aliases: set[str] | frozenset[str],
) -> str | None:
    """Return a forbidden artifact read by one direct or aliased call."""
    candidate: expr | None = None
    if _is_direct_reader(call.func, aliases, reader_aliases, values):
        candidate = call.args[0] if call.args else None
    elif isinstance(call.func, Attribute) and call.func.attr in {
        "read",
        "read_bytes",
        "read_text",
        "open",
    }:
        candidate = call.func.value
    else:
        candidate = _getattr_reader_base(call.func, aliases, values)
    if candidate is None:
        return None
    resolved = _constant_path(candidate, values, aliases)
    if resolved is not None and _is_forbidden_source_artifact(resolved):
        return resolved
    return None


def _is_direct_reader(
    value: expr,
    aliases: ArtifactAliases,
    reader_aliases: set[str] | frozenset[str],
    values: dict[str, str],
) -> bool:
    """Return whether one call target can directly open an artifact path."""
    if isinstance(value, Name):
        return value.id in aliases.reader_names | reader_aliases
    if _getattr_reader_base(value, aliases, values) is not None:
        return _getattr_reader_name(value, aliases, values) in {
            "open",
            "strict_json_path",
        }
    if not isinstance(value, Attribute):
        return False
    if value.attr not in {"open", "strict_json_path"}:
        return False
    return isinstance(value.value, Name) and value.value.id in (
        aliases.io_modules | aliases.os_modules | aliases.builtin_modules
    )


def _getattr_reader_name(
    value: expr,
    aliases: ArtifactAliases,
    values: dict[str, str],
) -> str | None:
    """Resolve a literal attribute name selected through ``getattr``."""
    if (
        not isinstance(value, Call)
        or not isinstance(value.func, Name)
        or value.func.id not in aliases.getattr_names
        or len(value.args) != 2
    ):
        return None
    return _constant_path(value.args[1], values, aliases)


def _getattr_reader_base(
    value: expr,
    aliases: ArtifactAliases,
    values: dict[str, str],
) -> expr | None:
    """Return the receiver of an indirect reader selected by ``getattr``."""
    name = _getattr_reader_name(value, aliases, values)
    if name not in {
        "open",
        "strict_json_path",
        "read",
        "read_bytes",
        "read_text",
    }:
        return None
    assert isinstance(value, Call)
    base = value.args[0]
    if name in {"open", "strict_json_path"}:
        if isinstance(base, Name) and base.id in (
            aliases.io_modules | aliases.os_modules | aliases.builtin_modules
        ):
            return base
        return None
    return base


def _is_forbidden_source_artifact(value: str) -> bool:
    """Return whether a normalized or absolute path names a design artifact."""
    normalized = value.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    return normalized in _FORBIDDEN_SOURCE_ARTIFACTS or tuple(
        candidate.parts[-2:]
    ) in {
        ("specs", "PATCH.md"),
        ("specs", "PATCH-agenda.md"),
    }


def _python_symbol_exists(path: Path, parts: tuple[str, ...]) -> bool:
    if not parts or any(not part for part in parts):
        return False
    try:
        tree = parse_python(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return False
    children = tuple(tree.body)
    for part in parts:
        found = next(
            (
                node
                for node in children
                if isinstance(node, (AsyncFunctionDef, ClassDef, FunctionDef))
                and node.name == part
            ),
            None,
        )
        if found is None:
            return False
        children = tuple(found.body)
    return True


def _load_mapping(path: Path, label: str) -> Mapping[str, object]:
    try:
        return mapping(strict_json_path(path), label)
    except (ContractGateError, StrictJsonError) as exc:
        raise PatchAcceptanceError(f"cannot read {label}: {exc}") from exc


def _header(
    payload: Mapping[str, object], label: str, *, schema_version: int = 1
) -> None:
    if (
        payload.get("schema_version") != schema_version
        or payload.get("feature") != _FEATURE
    ):
        raise PatchAcceptanceError(f"{label} header is invalid")


def _identifier(value: object, label: str) -> str:
    result = _string(value, label)
    if _IDENTIFIER_PATTERN.fullmatch(result) is None:
        raise PatchAcceptanceError(f"{label} is invalid")
    return result


def _symbol(value: object, label: str) -> str:
    """Return one Python symbol identifier used by executable evidence."""
    result = _string(value, label)
    if _SYMBOL_PATTERN.fullmatch(result) is None:
        raise PatchAcceptanceError(f"{label} is invalid")
    return result


def _identifier_list(value: object, label: str) -> tuple[str, ...]:
    values = tuple(
        _identifier(item, label) for item in object_list(value, label)
    )
    _unique(values, label)
    return values


def _line(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise PatchAcceptanceError(f"{label} is invalid")
    return value


def _phase(value: object, label: str) -> int:
    if type(value) is not int or value < 0 or value > _MAX_PHASE:
        raise PatchAcceptanceError(f"{label} is invalid")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PatchAcceptanceError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise PatchAcceptanceError(f"{label} is invalid")
    return result


def _hex_bytes(
    value: object, label: str, *, allow_empty: bool = False
) -> bytes:
    """Return one exact lower-case hexadecimal byte corpus."""
    if not isinstance(value, str) or (not value and not allow_empty):
        raise PatchAcceptanceError(f"{label} is invalid")
    text = value
    if len(text) % 2 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise PatchAcceptanceError(f"{label} is invalid")
    return bytes.fromhex(text)


def _safe_source_path(value: object) -> Path:
    return _safe_artifact_path(value, "source symbol path", suffix=".py")


def _safe_artifact_path(
    value: object, label: str, *, suffix: str | None = None
) -> Path:
    """Return one repository-relative artifact path."""
    text = _string(value, label)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in text
        or suffix is not None
        and path.suffix != suffix
    ):
        raise PatchAcceptanceError(f"{label} is invalid")
    return Path(*path.parts)


def _test_node(value: object) -> str:
    node = _string(value, "pytest node ID")
    relative = PurePosixPath(node.split("::", 1)[0])
    if (
        _NODE_PATTERN.fullmatch(node) is None
        or ".." in relative.parts
        or "\\" in node
    ):
        raise PatchAcceptanceError(f"invalid pytest node ID: {node}")
    return node


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _unique(values: Iterable[object], label: str) -> None:
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise PatchAcceptanceError(f"duplicate {label}")


def _exact_keys(
    payload: Mapping[str, object], expected: set[str], label: str
) -> None:
    if set(payload) != expected:
        raise PatchAcceptanceError(
            f"{label} has invalid keys: {sorted(set(payload) ^ expected)}"
        )


def _parse_args() -> Namespace:
    parser = ArgumentParser(description="Verify Phase 0 patch acceptance.")
    parser.add_argument("--through-phase", type=int, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=default_manifest_path()
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    return parser.parse_args()


def main() -> int:
    """Run the Phase 0 patch acceptance verifier."""
    args = _parse_args()
    try:
        manifest = verify_acceptance(
            args.manifest,
            repo_root=args.repo_root,
            through_phase=args.through_phase,
        )
    except PatchAcceptanceError as exc:
        print(f"patch acceptance failed: {exc}", file=stderr)
        return 1
    print(
        "patch acceptance passed: "
        f"through_phase={args.through_phase} nodes="
        f"{len(manifest.active_nodes(args.through_phase))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
