#!/usr/bin/env python
"""Provide deterministic dormant-patch contract test infrastructure.

This module is deliberately not a patch runtime. It models only the typed,
in-memory collaborators that Phase 0 uses to verify future lifecycle tests
without giving any production surface filesystem mutation authority.
"""

from asyncio import Event
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from html import escape
from json import dumps
from subprocess import PIPE, Popen
from sys import executable
from typing import Callable, Generic, NewType, Protocol, TypeVar

from contract_gate import strict_json_loads

PatchIdentifier = NewType("PatchIdentifier", str)
PatchRequestId = NewType("PatchRequestId", str)
PatchCallId = NewType("PatchCallId", str)
PatchPlanId = NewType("PatchPlanId", str)
PatchApprovalId = NewType("PatchApprovalId", str)
PatchOperationId = NewType("PatchOperationId", str)
PatchLineageId = NewType("PatchLineageId", str)
PatchStepId = NewType("PatchStepId", str)
PatchDomainId = NewType("PatchDomainId", str)
PatchContextId = NewType("PatchContextId", str)
PatchWorkspaceId = NewType("PatchWorkspaceId", str)
PatchEventId = NewType("PatchEventId", str)
PatchObserverId = NewType("PatchObserverId", str)
PatchDigestInput = NewType("PatchDigestInput", str)
PatchGrantId = NewType("PatchGrantId", str)
PatchLeaseId = NewType("PatchLeaseId", str)
PatchFenceId = NewType("PatchFenceId", str)
PatchCorrelationId = NewType("PatchCorrelationId", str)
PatchPath = NewType("PatchPath", str)
PatchHandleId = NewType("PatchHandleId", str)
PatchLockId = NewType("PatchLockId", str)
PatchStagingArtifactId = NewType("PatchStagingArtifactId", str)
PatchCapability = NewType("PatchCapability", str)
PatchPrincipalId = NewType("PatchPrincipalId", str)
PatchTenantId = NewType("PatchTenantId", str)
PatchRunId = NewType("PatchRunId", str)
PatchPolicyId = NewType("PatchPolicyId", str)
PatchBrokerId = NewType("PatchBrokerId", str)
PatchWorkspaceEntryId = NewType("PatchWorkspaceEntryId", str)
PatchStoreRecordId = NewType("PatchStoreRecordId", str)
PatchStoreRevision = NewType("PatchStoreRevision", int)
PatchArtifactDigest = NewType("PatchArtifactDigest", str)

PatchId = TypeVar("PatchId", bound=str)


def load_strict_json(source: str) -> object:
    """Decode JSON through the shared duplicate-preserving gate helper."""
    return strict_json_loads(source)


class GoldenCorpusCategory(str, Enum):
    """Name the complete executable Phase 0 golden corpus categories."""

    GRAMMAR = "grammar"
    PATH = "path"
    TEXT = "text"
    MATCHING = "matching"
    LINEAGE = "lineage"
    DIFF = "diff"
    FINGERPRINT = "fingerprint"
    RESULT = "result"
    ERROR = "error"
    EVENT = "event"
    REDACTION = "redaction"
    WIRE = "wire"


class ThreatCorpusIdentifier(str, Enum):
    """Name the complete executable Phase 0 adversarial corpus."""

    MALICIOUS_WORKSPACE_CONTENT = "malicious_workspace_content"
    TARGET_REPLACEMENT = "target_replacement"
    PROTOCOL_REPLAY = "protocol_replay"
    AUTHORITY_SWAP = "authority_swap"
    RENDERER_INJECTION = "renderer_injection"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass(frozen=True, kw_only=True, slots=True)
class CorpusOutcome:
    """Store the exact non-mutating result of one Phase 0 corpus case."""

    output_bytes: bytes
    outcome: str
    error: str


def execute_golden_corpus(
    category: GoldenCorpusCategory,
    input_bytes: bytes,
) -> CorpusOutcome:
    """Evaluate one frozen golden case without patch runtime authority."""
    match category:
        case GoldenCorpusCategory.GRAMMAR:
            try:
                decoded = load_strict_json(input_bytes.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_closed_json",
                    error=str(exc),
                )
            if not isinstance(decoded, dict) or set(decoded) != {"op"}:
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_closed_json",
                    error="closed request requires exactly one op member",
                )
            operation = decoded["op"]
            if operation not in {"edit", "apply"}:
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_closed_json",
                    error="closed request operation is invalid",
                )
            return CorpusOutcome(
                output_bytes=dumps(
                    {"op": operation}, separators=(",", ":")
                ).encode("utf-8"),
                outcome="accept_closed_json",
                error="none",
            )
        case GoldenCorpusCategory.PATH:
            try:
                logical_path = input_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_logical_path",
                    error="logical path must be UTF-8",
                )
            if (
                not logical_path
                or logical_path.startswith("/")
                or "\\" in logical_path
            ):
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_logical_path",
                    error="absolute or platform path is prohibited",
                )
            if any(
                part in {"", ".", ".."} for part in logical_path.split("/")
            ):
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_logical_path",
                    error="logical path has an ambiguous component",
                )
            return CorpusOutcome(
                output_bytes=input_bytes,
                outcome="accept_relative_path",
                error="none",
            )
        case GoldenCorpusCategory.TEXT:
            try:
                text = input_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_text",
                    error="text must be strict UTF-8",
                )
            if "\x00" in text or "\r" in text.replace("\r\n", ""):
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_text",
                    error=(
                        "text contains an unsupported control representation"
                    ),
                )
            if "\r\n" in text and "\n" in text.replace("\r\n", ""):
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_text",
                    error="text mixes logical newline representations",
                )
            return CorpusOutcome(
                output_bytes=input_bytes,
                outcome=(
                    "preserve_crlf_bytes"
                    if "\r\n" in text
                    else "preserve_lf_bytes"
                ),
                error="none",
            )
        case GoldenCorpusCategory.MATCHING:
            source, separator, needle = input_bytes.partition(b"\x00")
            if not separator or not needle:
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_no_exact_match",
                    error="matching input requires source and nonempty needle",
                )
            occurrences = source.count(needle)
            if occurrences != 1:
                detail = "missing" if occurrences == 0 else "ambiguous"
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_no_exact_match",
                    error=f"exact logical text match is {detail}",
                )
            offset = source.index(needle)
            return CorpusOutcome(
                output_bytes=f"match:{offset}:{len(needle)}".encode("ascii"),
                outcome="one_exact_match",
                error="none",
            )
        case GoldenCorpusCategory.LINEAGE:
            source, separator, destination = input_bytes.partition(b"->")
            if (
                not separator
                or not source
                or not destination
                or source == destination
            ):
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_lineage",
                    error="lineage requires distinct source and destination",
                )
            return CorpusOutcome(
                output_bytes=b"lineage:" + input_bytes,
                outcome="one_terminal_lineage",
                error="none",
            )
        case GoldenCorpusCategory.DIFF:
            diff_path, source, destination = _decode_triplet(input_bytes)
            if not diff_path or source == destination:
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_review_diff",
                    error="review diff requires one changed logical file",
                )
            return CorpusOutcome(
                output_bytes=(
                    b"--- a/"
                    + diff_path
                    + b"\n+++ b/"
                    + diff_path
                    + b"\n-"
                    + source
                    + b"\n+"
                    + destination
                    + b"\n"
                ),
                outcome="complete_review_diff",
                error="none",
            )
        case GoldenCorpusCategory.FINGERPRINT:
            return CorpusOutcome(
                output_bytes=sha256(input_bytes).hexdigest().encode("ascii"),
                outcome="stable_fingerprint",
                error="none",
            )
        case GoldenCorpusCategory.RESULT:
            status, separator, started = input_bytes.partition(b";")
            if separator != b";" or started not in {
                b"commit_started=true",
                b"commit_started=false",
            }:
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_result_projection",
                    error="result projection is malformed",
                )
            if (
                status == b"not_committed"
                and started != b"commit_started=false"
            ):
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_result_projection",
                    error="not_committed result cannot start commit",
                )
            if status not in {
                b"not_committed",
                b"committed",
                b"partially_committed",
                b"indeterminate",
            }:
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_result_projection",
                    error="result mutation state is invalid",
                )
            return CorpusOutcome(
                output_bytes=(
                    b'{"commit_started":'
                    + (b"true" if started.endswith(b"true") else b"false")
                    + b',"status":"'
                    + status
                    + b'"}'
                ),
                outcome="closed_result_projection",
                error="none",
            )
        case GoldenCorpusCategory.ERROR:
            error_map = {
                b"stale": (b"stale_plan", "stale plan requires a new request"),
                b"denied": (
                    b"policy_denied",
                    "policy denied the sealed request",
                ),
            }
            resolved = error_map.get(input_bytes)
            if resolved is None:
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_error_projection",
                    error="stable error input is unknown",
                )
            code, message = resolved
            return CorpusOutcome(
                output_bytes=b'{"error":"' + code + b'"}',
                outcome="closed_error_projection",
                error=message,
            )
        case GoldenCorpusCategory.EVENT:
            events = tuple(item for item in input_bytes.split(b">") if item)
            if not events or events[0] != b"request_received":
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_event_sequence",
                    error="event sequence must start with request_received",
                )
            if (
                events.count(b"request_completed") != 1
                or events[-1] != b"request_completed"
            ):
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_event_sequence",
                    error="event sequence requires one terminal completion",
                )
            return CorpusOutcome(
                output_bytes=b",".join(events),
                outcome="one_terminal_sequence",
                error="none",
            )
        case GoldenCorpusCategory.REDACTION:
            digest = sha256(input_bytes).hexdigest()[:12].encode("ascii")
            return CorpusOutcome(
                output_bytes=b"<redacted:" + digest + b">",
                outcome="audience_redacts_content",
                error="none",
            )
        case GoldenCorpusCategory.WIRE:
            try:
                decoded = load_strict_json(input_bytes.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_wire",
                    error=str(exc),
                )
            if not isinstance(decoded, dict) or decoded.get("v") != 1:
                return CorpusOutcome(
                    output_bytes=b"",
                    outcome="reject_wire",
                    error="wire envelope requires version 1",
                )
            return CorpusOutcome(
                output_bytes=dumps(
                    decoded, sort_keys=True, separators=(",", ":")
                ).encode("utf-8"),
                outcome="versioned_wire_round_trip",
                error="none",
            )


def execute_threat_corpus(
    identifier: ThreatCorpusIdentifier,
    setup_bytes: bytes,
    action_bytes: bytes,
) -> CorpusOutcome:
    """Contain one frozen adversarial case without workspace access."""
    assert setup_bytes and action_bytes
    match identifier:
        case ThreatCorpusIdentifier.MALICIOUS_WORKSPACE_CONTENT:
            if not setup_bytes.startswith(
                b"workspace:"
            ) or not action_bytes.startswith(b"read:"):
                return _invalid_threat_input(identifier)
            return CorpusOutcome(
                output_bytes=b"no-process-dispatch:"
                + sha256(setup_bytes + action_bytes)
                .hexdigest()[:12]
                .encode("ascii"),
                outcome="contained",
                error="workspace content is data, not executable authority",
            )
        case ThreatCorpusIdentifier.TARGET_REPLACEMENT:
            if not setup_bytes.startswith(
                b"root:"
            ) or not action_bytes.startswith(b"replace:"):
                return _invalid_threat_input(identifier)
            return CorpusOutcome(
                output_bytes=b"target-revalidation-required:"
                + sha256(action_bytes).hexdigest()[:12].encode("ascii"),
                outcome="contained",
                error="replaced target invalidates the request",
            )
        case ThreatCorpusIdentifier.PROTOCOL_REPLAY:
            if not setup_bytes.startswith(
                b"grant:"
            ) or not action_bytes.startswith(b"replay:"):
                return _invalid_threat_input(identifier)
            return CorpusOutcome(
                output_bytes=b"single-consumption:"
                + sha256(setup_bytes).hexdigest()[:12].encode("ascii"),
                outcome="contained",
                error="replayed grant has no second effect",
            )
        case ThreatCorpusIdentifier.AUTHORITY_SWAP:
            if not setup_bytes.startswith(b"binding:") or not action_bytes:
                return _invalid_threat_input(identifier)
            return CorpusOutcome(
                output_bytes=b"binding-mismatch:"
                + sha256(setup_bytes + b"\x00" + action_bytes)
                .hexdigest()[:12]
                .encode("ascii"),
                outcome="contained",
                error="authority binding mismatch",
            )
        case ThreatCorpusIdentifier.RENDERER_INJECTION:
            rendered = escape(action_bytes.decode("utf-8"), quote=True)
            return CorpusOutcome(
                output_bytes=rendered.encode("utf-8"),
                outcome="contained",
                error="renderer projection escaped untrusted content",
            )
        case ThreatCorpusIdentifier.RESOURCE_EXHAUSTION:
            _, separator, raw_limit = setup_bytes.partition(b":")
            if separator != b":" or not raw_limit.isdigit():
                return _invalid_threat_input(identifier)
            limit = int(raw_limit)
            if len(action_bytes) <= limit:
                return CorpusOutcome(
                    output_bytes=(
                        f"budget-accepted:{len(action_bytes)}".encode("ascii")
                    ),
                    outcome="contained",
                    error=(
                        "bounded corpus input remains within configured limit"
                    ),
                )
            return CorpusOutcome(
                output_bytes=(
                    f"budget-exceeded:{len(action_bytes)}>{limit}".encode(
                        "ascii"
                    )
                ),
                outcome="contained",
                error="bounded corpus input exceeds configured limit",
            )


def _decode_triplet(value: bytes) -> tuple[bytes, bytes, bytes]:
    """Decode one exact ``path|before|after`` golden input."""
    parts = value.split(b"|", 2)
    if len(parts) != 3:
        return b"", b"", b""
    return parts[0], parts[1], parts[2]


def _invalid_threat_input(identifier: ThreatCorpusIdentifier) -> CorpusOutcome:
    """Return a deterministic fail-closed threat corpus outcome."""
    return CorpusOutcome(
        output_bytes=identifier.value.encode("ascii"),
        outcome="rejected",
        error="threat corpus input is malformed",
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class ManualClock:
    """Represent a deterministic monotonic clock value."""

    tick: int

    def __post_init__(self) -> None:
        """Reject a clock that starts before its deterministic origin."""
        assert self.tick >= 0

    def advance(self, ticks: int = 1) -> "ManualClock":
        """Return a clock advanced by a positive number of ticks."""
        assert ticks > 0
        return ManualClock(tick=self.tick + ticks)


@dataclass(frozen=True, kw_only=True, slots=True)
class DeterministicFactory(Generic[PatchId]):
    """Return one typed identity and its next immutable factory state."""

    prefix: str
    constructor: Callable[[str], PatchId]
    next_ordinal: int = 0

    def __post_init__(self) -> None:
        """Reject an ambiguous identity factory configuration."""
        assert self.prefix and self.next_ordinal >= 0

    def issue(self) -> tuple[PatchId, "DeterministicFactory[PatchId]"]:
        """Issue one identity without mutable global state."""
        identifier = self.constructor(f"{self.prefix}-{self.next_ordinal:04d}")
        return identifier, DeterministicFactory(
            prefix=self.prefix,
            constructor=self.constructor,
            next_ordinal=self.next_ordinal + 1,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class PatchFactories:
    """Store every deterministic identity factory used by Phase 0 tests."""

    requests: DeterministicFactory[PatchRequestId]
    calls: DeterministicFactory[PatchCallId]
    plans: DeterministicFactory[PatchPlanId]
    approvals: DeterministicFactory[PatchApprovalId]
    operations: DeterministicFactory[PatchOperationId]
    lineages: DeterministicFactory[PatchLineageId]
    steps: DeterministicFactory[PatchStepId]
    domains: DeterministicFactory[PatchDomainId]
    contexts: DeterministicFactory[PatchContextId]
    workspaces: DeterministicFactory[PatchWorkspaceId]
    events: DeterministicFactory[PatchEventId]
    observers: DeterministicFactory[PatchObserverId]
    digest_inputs: DeterministicFactory[PatchDigestInput]
    grants: DeterministicFactory[PatchGrantId]
    leases: DeterministicFactory[PatchLeaseId]
    fences: DeterministicFactory[PatchFenceId]
    correlations: DeterministicFactory[PatchCorrelationId]

    @classmethod
    def create(cls) -> "PatchFactories":
        """Create the complete fixed set of deterministic factories."""
        return cls(
            requests=DeterministicFactory(
                prefix="request", constructor=PatchRequestId
            ),
            calls=DeterministicFactory(prefix="call", constructor=PatchCallId),
            plans=DeterministicFactory(prefix="plan", constructor=PatchPlanId),
            approvals=DeterministicFactory(
                prefix="approval", constructor=PatchApprovalId
            ),
            operations=DeterministicFactory(
                prefix="operation", constructor=PatchOperationId
            ),
            lineages=DeterministicFactory(
                prefix="lineage", constructor=PatchLineageId
            ),
            steps=DeterministicFactory(prefix="step", constructor=PatchStepId),
            domains=DeterministicFactory(
                prefix="domain", constructor=PatchDomainId
            ),
            contexts=DeterministicFactory(
                prefix="context", constructor=PatchContextId
            ),
            workspaces=DeterministicFactory(
                prefix="workspace", constructor=PatchWorkspaceId
            ),
            events=DeterministicFactory(
                prefix="event", constructor=PatchEventId
            ),
            observers=DeterministicFactory(
                prefix="observer", constructor=PatchObserverId
            ),
            digest_inputs=DeterministicFactory(
                prefix="digest-input", constructor=PatchDigestInput
            ),
            grants=DeterministicFactory(
                prefix="grant", constructor=PatchGrantId
            ),
            leases=DeterministicFactory(
                prefix="lease", constructor=PatchLeaseId
            ),
            fences=DeterministicFactory(
                prefix="fence", constructor=PatchFenceId
            ),
            correlations=DeterministicFactory(
                prefix="correlation", constructor=PatchCorrelationId
            ),
        )


class FaultLabel(str, Enum):
    """Name every frozen asynchronous Phase 0 test boundary."""

    LIFECYCLE_BEFORE = "lifecycle.before"
    LIFECYCLE_AFTER = "lifecycle.after"
    TARGET_BEFORE = "target.before"
    TARGET_AFTER = "target.after"
    STORE_BEFORE = "store.before"
    STORE_AFTER = "store.after"
    APPROVAL_BEFORE = "approval.before"
    APPROVAL_AFTER = "approval.after"
    COMMIT_BEFORE = "commit.before"
    COMMIT_AFTER = "commit.after"
    ARTIFACT_BEFORE = "artifact.before"
    ARTIFACT_AFTER = "artifact.after"
    CLEANUP_BEFORE = "cleanup.before"
    CLEANUP_AFTER = "cleanup.after"
    CANCELLATION_BEFORE = "cancellation.before"
    CANCELLATION_AFTER = "cancellation.after"
    PUBLICATION_BEFORE = "publication.before"
    PUBLICATION_AFTER = "publication.after"


_FROZEN_FAULT_LABELS = tuple(FaultLabel)


@dataclass(frozen=True, kw_only=True, slots=True)
class FaultBarrier:
    """Represent one named, manually released asynchronous test boundary."""

    label: FaultLabel
    entered: Event
    released: Event


@dataclass(frozen=True, kw_only=True, slots=True)
class FaultController:
    """Coordinate the closed set of asynchronous lifecycle barriers."""

    barriers: tuple[FaultBarrier, ...]

    @classmethod
    def create(
        cls,
        labels: tuple[FaultLabel, ...] | None = None,
    ) -> "FaultController":
        """Create barriers for the complete frozen label set only."""
        resolved_labels = _FROZEN_FAULT_LABELS if labels is None else labels
        assert resolved_labels == _FROZEN_FAULT_LABELS
        return cls(
            barriers=tuple(
                FaultBarrier(label=label, entered=Event(), released=Event())
                for label in resolved_labels
            )
        )

    async def arrive(
        self,
        label: FaultLabel,
        sentinel: "ResourceDepthSentinel | None" = None,
    ) -> "ResourceDepthSentinel":
        """Signal and wait at one explicitly configured barrier."""
        current = sentinel or ResourceDepthSentinel()
        checked = await current.at_await(AwaitBoundary.FAULT_WAIT)
        barrier = self._barrier(label)
        barrier.entered.set()
        await barrier.released.wait()
        return checked

    async def wait_until_entered(
        self,
        label: FaultLabel,
        sentinel: "ResourceDepthSentinel | None" = None,
    ) -> "ResourceDepthSentinel":
        """Wait until an asynchronous actor reaches one named barrier."""
        current = sentinel or ResourceDepthSentinel()
        checked = await current.at_await(AwaitBoundary.FAULT_WAIT)
        await self._barrier(label).entered.wait()
        return checked

    def release(self, label: FaultLabel) -> None:
        """Release one named barrier without advancing time."""
        self._barrier(label).released.set()

    def _barrier(self, label: FaultLabel) -> FaultBarrier:
        """Return the uniquely configured barrier for a frozen label."""
        matches = tuple(item for item in self.barriers if item.label == label)
        assert len(matches) == 1
        return matches[0]


class TargetTraceAction(str, Enum):
    """Name the complete closed scripted-target operation vocabulary."""

    NEGOTIATE_CAPABILITIES = "negotiate_capabilities"
    INSPECT = "inspect"
    OBSERVE_PRECONDITION = "observe_precondition"
    OPEN_HANDLE = "open_handle"
    CLOSE_HANDLE = "close_handle"
    ACQUIRE_LOCK = "acquire_lock"
    RELEASE_LOCK = "release_lock"
    STAGE_ARTIFACT = "stage_artifact"
    CLEAN_STAGING_ARTIFACT = "clean_staging_artifact"
    NAMESPACE_MUTATION = "namespace_mutation"
    COMMIT_STEP = "commit_step"
    VERIFY = "verify"


class TargetTraceSubjectKind(str, Enum):
    """Name the exact typed subject category for one target action."""

    NONE = "none"
    CAPABILITY = "capability"
    PATH = "path"
    HANDLE = "handle"
    LOCK = "lock"
    ARTIFACT = "artifact"
    STEP = "step"


TargetTraceSubjectValue = (
    PatchCapability
    | PatchHandleId
    | PatchLockId
    | PatchPath
    | PatchStagingArtifactId
    | PatchStepId
)


@dataclass(frozen=True, kw_only=True, slots=True)
class TargetTraceSubject:
    """Bind one trace subject to a closed kind and typed identity value."""

    kind: TargetTraceSubjectKind
    value: TargetTraceSubjectValue | None = None

    def __post_init__(self) -> None:
        """Reject an empty or mismatched trace subject category."""
        assert (self.kind is TargetTraceSubjectKind.NONE) == (
            self.value is None
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class TargetTrace:
    """Record one closed target action, subject, and namespace fact."""

    action: TargetTraceAction
    subject: TargetTraceSubject
    workspace_namespace_mutation: bool
    await_receipt: "AwaitReceipt"


@dataclass(frozen=True, kw_only=True, slots=True)
class ScriptedMutationTarget:
    """Model a target protocol without filesystem access or mutation."""

    capabilities: tuple[PatchCapability, ...] = ()
    trace: tuple[TargetTrace, ...] = ()
    sentinel: "ResourceDepthSentinel" = field(
        default_factory=lambda: ResourceDepthSentinel()
    )
    faults: FaultController | None = None
    fault_label: FaultLabel | None = None

    def __post_init__(self) -> None:
        """Reject non-canonical scripted capability responses."""
        assert tuple(sorted(self.capabilities)) == self.capabilities
        assert len(self.capabilities) == len(set(self.capabilities))

    async def negotiate_capabilities(
        self,
    ) -> tuple[tuple[PatchCapability, ...], "ScriptedMutationTarget"]:
        """Record one capability negotiation without a target handshake."""
        receipt, sentinel = await self._await(AwaitBoundary.TARGET_NEGOTIATION)
        return self.capabilities, self._record(
            TargetTraceAction.NEGOTIATE_CAPABILITIES,
            TargetTraceSubject(kind=TargetTraceSubjectKind.NONE),
            receipt,
            sentinel=sentinel,
        )

    async def inspect(
        self,
        path: PatchPath,
    ) -> tuple[TargetTrace, "ScriptedMutationTarget"]:
        """Record one read-only inspection in the scripted trace."""
        receipt, sentinel = await self._await(AwaitBoundary.TARGET_INSPECTION)
        return self._record_with_trace(
            TargetTraceAction.INSPECT,
            TargetTraceSubject(kind=TargetTraceSubjectKind.PATH, value=path),
            receipt,
            sentinel=sentinel,
        )

    async def observe_precondition(
        self,
        path: PatchPath,
    ) -> tuple[TargetTrace, "ScriptedMutationTarget"]:
        """Record a non-disclosing precondition observation."""
        receipt, sentinel = await self._await(
            AwaitBoundary.TARGET_PRECONDITION
        )
        return self._record_with_trace(
            TargetTraceAction.OBSERVE_PRECONDITION,
            TargetTraceSubject(kind=TargetTraceSubjectKind.PATH, value=path),
            receipt,
            sentinel=sentinel,
        )

    async def open_handle(
        self,
        handle: PatchHandleId,
    ) -> tuple[TargetTrace, "ScriptedMutationTarget"]:
        """Record one target-handle acquisition without opening a file."""
        receipt, sentinel = await self._await(AwaitBoundary.TARGET_HANDLE_OPEN)
        return self._record_with_trace(
            TargetTraceAction.OPEN_HANDLE,
            TargetTraceSubject(
                kind=TargetTraceSubjectKind.HANDLE,
                value=handle,
            ),
            receipt,
            sentinel=sentinel.acquire(ResourceOwner.TARGET_HANDLE),
        )

    async def close_handle(
        self,
        handle: PatchHandleId,
    ) -> tuple[TargetTrace, "ScriptedMutationTarget"]:
        """Record one target-handle release without closing a file."""
        receipt, sentinel = await self._await(
            AwaitBoundary.TARGET_HANDLE_CLOSE
        )
        sentinel = await self._fault_wait(sentinel)
        return self._record_with_trace(
            TargetTraceAction.CLOSE_HANDLE,
            TargetTraceSubject(
                kind=TargetTraceSubjectKind.HANDLE,
                value=handle,
            ),
            receipt,
            sentinel=sentinel.release(ResourceOwner.TARGET_HANDLE),
        )

    async def acquire_lock(
        self,
        lock: PatchLockId,
    ) -> tuple[TargetTrace, "ScriptedMutationTarget"]:
        """Record one target-lock acquisition without process access."""
        receipt, sentinel = await self._await(
            AwaitBoundary.TARGET_LOCK_ACQUIRE
        )
        return self._record_with_trace(
            TargetTraceAction.ACQUIRE_LOCK,
            TargetTraceSubject(kind=TargetTraceSubjectKind.LOCK, value=lock),
            receipt,
            sentinel=sentinel.acquire(ResourceOwner.COORDINATOR_LEASE),
        )

    async def release_lock(
        self,
        lock: PatchLockId,
    ) -> tuple[TargetTrace, "ScriptedMutationTarget"]:
        """Record one target-lock release without synchronizing a process."""
        receipt, sentinel = await self._await(
            AwaitBoundary.TARGET_LOCK_RELEASE
        )
        return self._record_with_trace(
            TargetTraceAction.RELEASE_LOCK,
            TargetTraceSubject(kind=TargetTraceSubjectKind.LOCK, value=lock),
            receipt,
            sentinel=sentinel.release(ResourceOwner.COORDINATOR_LEASE),
        )

    async def stage_artifact(
        self,
        artifact: PatchStagingArtifactId,
    ) -> tuple[TargetTrace, "ScriptedMutationTarget"]:
        """Record staging ownership without creating a workspace entry."""
        receipt, sentinel = await self._await(AwaitBoundary.TARGET_STAGE)
        return self._record_with_trace(
            TargetTraceAction.STAGE_ARTIFACT,
            TargetTraceSubject(
                kind=TargetTraceSubjectKind.ARTIFACT,
                value=artifact,
            ),
            receipt,
            sentinel=sentinel.acquire(ResourceOwner.STAGING_RESOURCE),
        )

    async def clean_staging_artifact(
        self,
        artifact: PatchStagingArtifactId,
    ) -> tuple[TargetTrace, "ScriptedMutationTarget"]:
        """Record staging cleanup without deleting a workspace entry."""
        receipt, sentinel = await self._await(AwaitBoundary.TARGET_CLEANUP)
        return self._record_with_trace(
            TargetTraceAction.CLEAN_STAGING_ARTIFACT,
            TargetTraceSubject(
                kind=TargetTraceSubjectKind.ARTIFACT,
                value=artifact,
            ),
            receipt,
            sentinel=sentinel.release(ResourceOwner.STAGING_RESOURCE),
        )

    async def record_namespace_mutation(
        self,
        path: PatchPath,
    ) -> tuple[TargetTrace, "ScriptedMutationTarget"]:
        """Record a namespace mutation without writing a workspace."""
        receipt, sentinel = await self._await(
            AwaitBoundary.TARGET_NAMESPACE_MUTATION
        )
        return self._record_with_trace(
            TargetTraceAction.NAMESPACE_MUTATION,
            TargetTraceSubject(kind=TargetTraceSubjectKind.PATH, value=path),
            receipt,
            sentinel=sentinel,
            workspace_namespace_mutation=True,
        )

    async def commit_step(
        self,
        step: PatchStepId,
    ) -> tuple[TargetTrace, "ScriptedMutationTarget"]:
        """Record one requested-effect commit step without mutation."""
        receipt, sentinel = await self._await(
            AwaitBoundary.TARGET_COMMIT,
            acquire=ResourceOwner.TARGET_WORKER,
        )
        sentinel = await self._fault_wait(sentinel)
        return self._record_with_trace(
            TargetTraceAction.COMMIT_STEP,
            TargetTraceSubject(kind=TargetTraceSubjectKind.STEP, value=step),
            receipt,
            sentinel=sentinel.release(ResourceOwner.TARGET_WORKER),
        )

    async def verify(
        self,
        path: PatchPath,
    ) -> tuple[TargetTrace, "ScriptedMutationTarget"]:
        """Record structural verification without reading a workspace."""
        receipt, sentinel = await self._await(
            AwaitBoundary.TARGET_VERIFICATION
        )
        return self._record_with_trace(
            TargetTraceAction.VERIFY,
            TargetTraceSubject(kind=TargetTraceSubjectKind.PATH, value=path),
            receipt,
            sentinel=sentinel,
        )

    async def _await(
        self,
        boundary: "AwaitBoundary",
        *,
        acquire: "ResourceOwner | None" = None,
    ) -> tuple["AwaitReceipt", "ResourceDepthSentinel"]:
        """Record one boundary using this target's current ownership state."""
        current = (
            self.sentinel
            if acquire is None
            else self.sentinel.acquire(acquire)
        )
        checked = await current.at_await(boundary)
        return checked.receipts[-1], checked

    async def _fault_wait(
        self,
        sentinel: "ResourceDepthSentinel",
    ) -> "ResourceDepthSentinel":
        """Wait at this target's configured fault barrier with live owners."""
        if self.faults is None or self.fault_label is None:
            return sentinel
        return await self.faults.arrive(self.fault_label, sentinel)

    def _record_with_trace(
        self,
        action: TargetTraceAction,
        subject: TargetTraceSubject,
        await_receipt: "AwaitReceipt",
        *,
        sentinel: "ResourceDepthSentinel",
        workspace_namespace_mutation: bool = False,
    ) -> tuple[TargetTrace, "ScriptedMutationTarget"]:
        """Return one trace record and a target with that record appended."""
        record = TargetTrace(
            action=action,
            subject=subject,
            workspace_namespace_mutation=workspace_namespace_mutation,
            await_receipt=await_receipt,
        )
        return record, self._record(
            action,
            subject,
            await_receipt,
            sentinel=sentinel,
            workspace_namespace_mutation=workspace_namespace_mutation,
        )

    def _record(
        self,
        action: TargetTraceAction,
        subject: TargetTraceSubject,
        await_receipt: "AwaitReceipt",
        *,
        sentinel: "ResourceDepthSentinel",
        workspace_namespace_mutation: bool = False,
    ) -> "ScriptedMutationTarget":
        """Return a target with one appended immutable trace record."""
        return ScriptedMutationTarget(
            capabilities=self.capabilities,
            trace=(
                *self.trace,
                TargetTrace(
                    action=action,
                    subject=subject,
                    workspace_namespace_mutation=workspace_namespace_mutation,
                    await_receipt=await_receipt,
                ),
            ),
            sentinel=sentinel,
            faults=self.faults,
            fault_label=self.fault_label,
        )


class ApprovalDecision(str, Enum):
    """Name the closed review outcomes a scripted broker can issue."""

    APPROVE = "approve"
    DENY = "deny"
    UNAVAILABLE = "unavailable"


class ApprovalOutcomeKind(str, Enum):
    """Name the deterministic result of review or grant consumption."""

    APPROVED = "approved"
    DENIED = "denied"
    UNAVAILABLE = "unavailable"
    EXPIRED = "expired"
    BINDING_MISMATCH = "binding_mismatch"
    REPLAYED = "replayed"


class ApprovalBindingMismatch(str, Enum):
    """Name one exact binding mismatch before a grant is issued or consumed."""

    PLAN = "plan"
    PRINCIPAL = "principal"
    TENANT = "tenant"
    RUN = "run"
    CONTEXT = "context"
    WORKSPACE = "workspace"
    POLICY = "policy"
    BROKER = "broker"
    QUORUM = "quorum"


@dataclass(frozen=True, kw_only=True, slots=True)
class ApprovalBinding:
    """Bind one approval script to its complete immutable review scope."""

    plan_id: PatchPlanId
    principal_id: PatchPrincipalId
    tenant_id: PatchTenantId
    run_id: PatchRunId
    context_id: PatchContextId
    workspace_id: PatchWorkspaceId
    policy_id: PatchPolicyId
    broker_id: PatchBrokerId
    quorum: int

    def __post_init__(self) -> None:
        """Reject a review binding without a positive reviewer quorum."""
        assert self.quorum > 0


@dataclass(frozen=True, kw_only=True, slots=True)
class PatchApprovalGrant:
    """Store one typed expiring grant bound to an exact approval scope."""

    identifier: PatchGrantId
    binding: ApprovalBinding
    issued_tick: int
    expires_tick: int

    def __post_init__(self) -> None:
        """Reject a grant whose expiration predates its issuance."""
        assert self.issued_tick >= 0 and self.expires_tick >= self.issued_tick


@dataclass(frozen=True, kw_only=True, slots=True)
class ApprovalOutcome:
    """Store a typed broker decision without mutable approval authority."""

    kind: ApprovalOutcomeKind
    grant: PatchApprovalGrant | None = None
    mismatch: ApprovalBindingMismatch | None = None

    def __post_init__(self) -> None:
        """Keep grant and mismatch fields consistent with the outcome kind."""
        if self.kind is ApprovalOutcomeKind.APPROVED:
            assert self.grant is not None and self.mismatch is None
        elif self.kind is ApprovalOutcomeKind.BINDING_MISMATCH:
            assert self.grant is None and self.mismatch is not None
        else:
            assert self.grant is None and self.mismatch is None


@dataclass(frozen=True, kw_only=True, slots=True)
class GrantConsumption:
    """Store one deterministically arbitrated concurrent grant consumer."""

    observer_id: PatchObserverId
    outcome: ApprovalOutcome


@dataclass(frozen=True, kw_only=True, slots=True)
class ScriptedApprovalBroker:
    """Script immutable plan-bound approval and one-time grant consumption."""

    binding: ApprovalBinding
    decision: ApprovalDecision
    grant: PatchApprovalGrant | None = None
    delay_label: FaultLabel | None = None
    calls: tuple[ApprovalBinding, ...] = ()
    consumed_grants: tuple[PatchGrantId, ...] = ()
    await_receipts: tuple["AwaitReceipt", ...] = ()
    sentinel: "ResourceDepthSentinel" = field(
        default_factory=lambda: ResourceDepthSentinel()
    )

    def __post_init__(self) -> None:
        """Reject a broker script whose grant is not bound to its decision."""
        if self.decision is ApprovalDecision.APPROVE:
            assert (
                self.grant is not None and self.grant.binding == self.binding
            )
        else:
            assert self.grant is None
        assert len(self.consumed_grants) == len(set(self.consumed_grants))
        if self.grant is None:
            assert not self.consumed_grants
        else:
            assert all(
                identifier == self.grant.identifier
                for identifier in self.consumed_grants
            )

    async def decide(
        self,
        binding: ApprovalBinding,
        clock: ManualClock,
        *,
        faults: FaultController | None = None,
    ) -> tuple[ApprovalOutcome, "ScriptedApprovalBroker"]:
        """Return one scripted decision after exact binding validation."""
        receipt, sentinel = await self._approval_await(
            AwaitBoundary.APPROVAL_DECISION
        )
        next_broker = self._record_call(binding, receipt, sentinel)
        mismatch = _approval_binding_mismatch(self.binding, binding)
        if mismatch is not None:
            return (
                ApprovalOutcome(
                    kind=ApprovalOutcomeKind.BINDING_MISMATCH,
                    mismatch=mismatch,
                ),
                next_broker,
            )
        if self.delay_label is not None:
            assert faults is not None
            next_broker = next_broker._with_sentinel(
                await faults.arrive(self.delay_label, next_broker.sentinel)
            )
        if self.grant is not None and clock.tick >= self.grant.expires_tick:
            return (
                ApprovalOutcome(kind=ApprovalOutcomeKind.EXPIRED),
                next_broker,
            )
        if self.decision is ApprovalDecision.APPROVE:
            assert self.grant is not None
            return (
                ApprovalOutcome(
                    kind=ApprovalOutcomeKind.APPROVED,
                    grant=self.grant,
                ),
                next_broker,
            )
        if self.decision is ApprovalDecision.DENY:
            return (
                ApprovalOutcome(kind=ApprovalOutcomeKind.DENIED),
                next_broker,
            )
        return (
            ApprovalOutcome(kind=ApprovalOutcomeKind.UNAVAILABLE),
            next_broker,
        )

    async def consume(
        self,
        grant: PatchApprovalGrant,
        binding: ApprovalBinding,
        clock: ManualClock,
    ) -> tuple[ApprovalOutcome, "ScriptedApprovalBroker"]:
        """Consume one matching grant once without mutating this broker."""
        receipt, sentinel = await self._approval_await(
            AwaitBoundary.APPROVAL_CONSUME
        )
        current = self._record_await_receipt(receipt, sentinel)
        mismatch = _approval_binding_mismatch(self.binding, binding)
        if mismatch is not None:
            return (
                ApprovalOutcome(
                    kind=ApprovalOutcomeKind.BINDING_MISMATCH,
                    mismatch=mismatch,
                ),
                current,
            )
        if self.grant != grant:
            return ApprovalOutcome(kind=ApprovalOutcomeKind.DENIED), current
        if clock.tick >= grant.expires_tick:
            return ApprovalOutcome(kind=ApprovalOutcomeKind.EXPIRED), current
        if grant.identifier in self.consumed_grants:
            return ApprovalOutcome(kind=ApprovalOutcomeKind.REPLAYED), current
        return (
            ApprovalOutcome(kind=ApprovalOutcomeKind.APPROVED, grant=grant),
            ScriptedApprovalBroker(
                binding=self.binding,
                decision=self.decision,
                grant=self.grant,
                delay_label=self.delay_label,
                calls=current.calls,
                consumed_grants=(*self.consumed_grants, grant.identifier),
                await_receipts=current.await_receipts,
                sentinel=current.sentinel,
            ),
        )

    async def consume_concurrently(
        self,
        grant: PatchApprovalGrant,
        binding: ApprovalBinding,
        clock: ManualClock,
        observers: tuple[PatchObserverId, ...],
    ) -> tuple[tuple[GrantConsumption, ...], "ScriptedApprovalBroker"]:
        """Arbitrate simultaneous consumers in one stable observer order."""
        assert observers and tuple(sorted(observers)) == observers
        assert len(observers) == len(set(observers))
        receipt, sentinel = await self._approval_await(
            AwaitBoundary.APPROVAL_CONCURRENT_CONSUME
        )
        broker = self._record_await_receipt(receipt, sentinel)
        consumptions: list[GrantConsumption] = []
        for observer_id in observers:
            outcome, broker = await broker.consume(grant, binding, clock)
            consumptions.append(
                GrantConsumption(observer_id=observer_id, outcome=outcome)
            )
        return tuple(consumptions), broker

    async def _approval_await(
        self,
        boundary: "AwaitBoundary",
    ) -> tuple["AwaitReceipt", "ResourceDepthSentinel"]:
        """Record an approval operation with its transient wait owner."""
        checked = await self.sentinel.acquire(
            ResourceOwner.APPROVAL_WAIT
        ).at_await(boundary)
        return (
            checked.receipts[-1],
            checked.release(ResourceOwner.APPROVAL_WAIT),
        )

    def _record_call(
        self,
        binding: ApprovalBinding,
        receipt: "AwaitReceipt",
        sentinel: "ResourceDepthSentinel",
    ) -> "ScriptedApprovalBroker":
        """Return this script with one immutable review call appended."""
        return ScriptedApprovalBroker(
            binding=self.binding,
            decision=self.decision,
            grant=self.grant,
            delay_label=self.delay_label,
            calls=(*self.calls, binding),
            consumed_grants=self.consumed_grants,
            await_receipts=(*self.await_receipts, receipt),
            sentinel=sentinel,
        )

    def _record_await_receipt(
        self,
        receipt: "AwaitReceipt",
        sentinel: "ResourceDepthSentinel",
    ) -> "ScriptedApprovalBroker":
        """Return this immutable broker with one sentinel-owned receipt."""
        return ScriptedApprovalBroker(
            binding=self.binding,
            decision=self.decision,
            grant=self.grant,
            delay_label=self.delay_label,
            calls=self.calls,
            consumed_grants=self.consumed_grants,
            await_receipts=(*self.await_receipts, receipt),
            sentinel=sentinel,
        )

    def _with_sentinel(
        self,
        sentinel: "ResourceDepthSentinel",
    ) -> "ScriptedApprovalBroker":
        """Return this immutable broker with its current sentinel replaced."""
        return ScriptedApprovalBroker(
            binding=self.binding,
            decision=self.decision,
            grant=self.grant,
            delay_label=self.delay_label,
            calls=self.calls,
            consumed_grants=self.consumed_grants,
            await_receipts=self.await_receipts,
            sentinel=sentinel,
        )


class WorkspaceEntryType(str, Enum):
    """Name the exact entry types represented by a workspace oracle."""

    DIRECTORY = "directory"
    FILE = "file"
    SYMLINK = "symlink"


@dataclass(frozen=True, kw_only=True, slots=True)
class WorkspaceSecurityMetadata:
    """Store one supported security-relevant metadata fact exactly."""

    name: str
    value: str

    def __post_init__(self) -> None:
        """Reject an unnamed or empty security metadata fact."""
        assert self.name and self.value


@dataclass(frozen=True, kw_only=True, slots=True)
class WorkspaceEntry:
    """Store one recursive logical workspace entry without disk access."""

    name: str
    entry_type: WorkspaceEntryType
    content: bytes
    symlink_target: str | None
    link_count: int
    identity: PatchWorkspaceEntryId
    mode: int
    security_metadata: tuple[WorkspaceSecurityMetadata, ...] = ()
    children: tuple["WorkspaceEntry", ...] = ()

    def __post_init__(self) -> None:
        """Reject an entry whose representation does not match its type."""
        assert (
            self.link_count > 0 and self.identity and 0 <= self.mode <= 0o7777
        )
        assert "/" not in self.name and self.name not in {".", ".."}
        assert tuple(
            sorted((item.name, item.value) for item in self.security_metadata)
        ) == tuple((item.name, item.value) for item in self.security_metadata)
        assert len({item.name for item in self.security_metadata}) == len(
            self.security_metadata
        )
        assert tuple(sorted(item.name for item in self.children)) == tuple(
            item.name for item in self.children
        )
        assert len({item.name for item in self.children}) == len(self.children)
        if self.entry_type is WorkspaceEntryType.DIRECTORY:
            assert self.content == b"" and self.symlink_target is None
        elif self.entry_type is WorkspaceEntryType.FILE:
            assert self.symlink_target is None and not self.children
        else:
            assert (
                self.content == b""
                and self.symlink_target
                and not self.children
            )


@dataclass(frozen=True, kw_only=True, slots=True)
class ArtifactNamespace:
    """Store one target-private artifact tree distinct from the workspace."""

    name: str
    root: WorkspaceEntry

    def __post_init__(self) -> None:
        """Reject an unnamed artifact namespace or a non-directory root."""
        assert (
            self.name and self.root.entry_type is WorkspaceEntryType.DIRECTORY
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class WorkspaceOracle:
    """Capture a recursive workspace, canaries, and artifact namespaces."""

    root: WorkspaceEntry
    outside_root_canaries: tuple[WorkspaceEntry, ...] = ()
    artifact_namespaces: tuple[ArtifactNamespace, ...] = ()

    def __post_init__(self) -> None:
        """Reject an oracle without a canonical root and side inventories."""
        assert self.root.name == ""
        assert self.root.entry_type is WorkspaceEntryType.DIRECTORY
        assert tuple(
            sorted(item.name for item in self.outside_root_canaries)
        ) == tuple(item.name for item in self.outside_root_canaries)
        assert len({item.name for item in self.outside_root_canaries}) == len(
            self.outside_root_canaries
        )
        assert tuple(
            sorted(item.name for item in self.artifact_namespaces)
        ) == tuple(item.name for item in self.artifact_namespaces)
        assert len({item.name for item in self.artifact_namespaces}) == len(
            self.artifact_namespaces
        )

    def digest(self) -> str:
        """Return a stable digest over every recursive protected fact."""
        payload = b"workspace-oracle-v1" + _workspace_entry_bytes(self.root)
        payload += b"canaries" + b"".join(
            _workspace_entry_bytes(entry)
            for entry in self.outside_root_canaries
        )
        payload += b"artifacts" + b"".join(
            _encoded_text(namespace.name)
            + _workspace_entry_bytes(namespace.root)
            for namespace in self.artifact_namespaces
        )
        return sha256(payload).hexdigest()

    def equals(self, other: "WorkspaceOracle") -> bool:
        """Return exact equality for two immutable oracle snapshots."""
        return self == other


def has_zero_write_evidence(
    before: WorkspaceOracle,
    after: WorkspaceOracle,
    target: ScriptedMutationTarget,
) -> bool:
    """Return whether both required precommit zero-write oracles agree."""
    return before.equals(after) and not any(
        record.workspace_namespace_mutation for record in target.trace
    )


def _approval_binding_mismatch(
    expected: ApprovalBinding,
    observed: ApprovalBinding,
) -> ApprovalBindingMismatch | None:
    """Return the first stable mismatch between two approval bindings."""
    for mismatch, expected_value, observed_value in (
        (ApprovalBindingMismatch.PLAN, expected.plan_id, observed.plan_id),
        (
            ApprovalBindingMismatch.PRINCIPAL,
            expected.principal_id,
            observed.principal_id,
        ),
        (
            ApprovalBindingMismatch.TENANT,
            expected.tenant_id,
            observed.tenant_id,
        ),
        (ApprovalBindingMismatch.RUN, expected.run_id, observed.run_id),
        (
            ApprovalBindingMismatch.CONTEXT,
            expected.context_id,
            observed.context_id,
        ),
        (
            ApprovalBindingMismatch.WORKSPACE,
            expected.workspace_id,
            observed.workspace_id,
        ),
        (
            ApprovalBindingMismatch.POLICY,
            expected.policy_id,
            observed.policy_id,
        ),
        (
            ApprovalBindingMismatch.BROKER,
            expected.broker_id,
            observed.broker_id,
        ),
        (ApprovalBindingMismatch.QUORUM, expected.quorum, observed.quorum),
    ):
        if expected_value != observed_value:
            return mismatch
    return None


def _encoded_text(value: str) -> bytes:
    """Return an unambiguous UTF-8 length-prefixed textual field."""
    return _encoded_bytes(value.encode("utf-8"))


def _encoded_bytes(value: bytes) -> bytes:
    """Return an unambiguous length-prefixed byte field."""
    return len(value).to_bytes(8, "big") + value


def _workspace_entry_bytes(entry: WorkspaceEntry) -> bytes:
    """Return one recursive canonical entry encoding for an oracle digest."""
    payload = b"entry"
    payload += _encoded_text(entry.name)
    payload += _encoded_text(entry.entry_type.value)
    payload += _encoded_bytes(entry.content)
    payload += _encoded_text(entry.symlink_target or "")
    payload += _encoded_text(str(entry.link_count))
    payload += _encoded_text(entry.identity)
    payload += _encoded_text(str(entry.mode))
    payload += b"metadata" + b"".join(
        _encoded_text(item.name) + _encoded_text(item.value)
        for item in entry.security_metadata
    )
    payload += b"children" + b"".join(
        _workspace_entry_bytes(child) for child in entry.children
    )
    return payload


class StoreBackend(str, Enum):
    """Name the future storage backends sharing one conformance suite."""

    IN_MEMORY = "in_memory"
    POSTGRESQL = "postgresql"


class StoreOperation(str, Enum):
    """Name the closed operations future patch stores must implement."""

    READ = "read"
    CREATE = "create"
    COMPARE_AND_SET = "compare_and_set"
    CLOSE = "close"


@dataclass(frozen=True, kw_only=True, slots=True)
class PatchStoreRecord:
    """Store one immutable record used by a future patch-store protocol."""

    identifier: PatchStoreRecordId
    revision: PatchStoreRevision
    digest_input: PatchDigestInput

    def __post_init__(self) -> None:
        """Reject a record with a negative optimistic-concurrency revision."""
        assert self.revision >= 0 and self.identifier and self.digest_input


class PatchStoreProtocol(Protocol):
    """Describe the strictly typed async boundary for future patch stores."""

    async def read(
        self,
        identifier: PatchStoreRecordId,
    ) -> PatchStoreRecord | None:
        """Return one record without granting mutation authority."""

    async def create(self, record: PatchStoreRecord) -> bool:
        """Create one absent record and report whether it was installed."""

    async def compare_and_set(
        self,
        expected: PatchStoreRecord,
        replacement: PatchStoreRecord,
    ) -> bool:
        """Replace one record only when its exact revision still matches."""

    async def close(self) -> None:
        """Close this store boundary before its owner exits."""


@dataclass(frozen=True, kw_only=True, slots=True)
class StoreConformanceCase:
    """Store one backend-neutral future-store conformance expectation."""

    identifier: str
    operation: StoreOperation
    expected_result: bool | None

    def __post_init__(self) -> None:
        """Reject an unnamed or underspecified store conformance case."""
        assert self.identifier
        if self.operation is StoreOperation.READ:
            assert self.expected_result is None
        else:
            assert type(self.expected_result) is bool


@dataclass(frozen=True, kw_only=True, slots=True)
class StoreConformanceSuite:
    """Define reusable cases for future in-memory and PostgreSQL stores."""

    backends: tuple[StoreBackend, ...]
    cases: tuple[StoreConformanceCase, ...]

    def __post_init__(self) -> None:
        """Reject an incomplete or non-deterministic store-suite inventory."""
        assert self.backends == tuple(StoreBackend)
        assert self.cases
        assert tuple(sorted(case.identifier for case in self.cases)) == tuple(
            case.identifier for case in self.cases
        )
        assert len({case.identifier for case in self.cases}) == len(self.cases)

    @classmethod
    def create(cls) -> "StoreConformanceSuite":
        """Create the frozen future-store conformance case inventory."""
        return cls(
            backends=tuple(StoreBackend),
            cases=(
                StoreConformanceCase(
                    identifier="close_owned_boundary",
                    operation=StoreOperation.CLOSE,
                    expected_result=True,
                ),
                StoreConformanceCase(
                    identifier="compare_and_set_conflict",
                    operation=StoreOperation.COMPARE_AND_SET,
                    expected_result=False,
                ),
                StoreConformanceCase(
                    identifier="compare_and_set_success",
                    operation=StoreOperation.COMPARE_AND_SET,
                    expected_result=True,
                ),
                StoreConformanceCase(
                    identifier="create_absent_record",
                    operation=StoreOperation.CREATE,
                    expected_result=True,
                ),
                StoreConformanceCase(
                    identifier="read_absent_record",
                    operation=StoreOperation.READ,
                    expected_result=None,
                ),
            ),
        )


class TargetProfileKind(str, Enum):
    """Name the contexts that must share future target conformance cases."""

    SCRIPTED = "scripted"
    LOCAL = "local"
    SANDBOX = "sandbox"
    CONTAINER = "container"


@dataclass(frozen=True, kw_only=True, slots=True)
class TargetConformanceProfile:
    """Describe one target-profile capability state without target authority.

    This profile has no target authority.
    """

    kind: TargetProfileKind
    capable: bool
    required_capabilities: tuple[PatchCapability, ...] = ()

    def __post_init__(self) -> None:
        """Reject an incapable profile that advertises target capabilities."""
        assert tuple(sorted(self.required_capabilities)) == (
            self.required_capabilities
        )
        assert len(self.required_capabilities) == len(
            set(self.required_capabilities)
        )
        if not self.capable:
            assert not self.required_capabilities


class TargetConformanceTarget(Protocol):
    """Describe the read-only handshake used by target conformance tests."""

    async def negotiate_capabilities(
        self,
    ) -> tuple[
        tuple[PatchCapability, ...],
        "TargetConformanceTarget",
    ]:
        """Return target capabilities and a successor without mutation."""


class TargetConformanceFactory(Protocol):
    """Describe a future factory for each target-profile test context."""

    async def create(
        self,
        profile: TargetConformanceProfile,
    ) -> TargetConformanceTarget | None:
        """Return a target only for a profile that can prove capability."""


@dataclass(frozen=True, kw_only=True, slots=True)
class TargetConformanceResult:
    """Store one profile result from the shared target conformance runner."""

    kind: TargetProfileKind
    capable: bool
    capabilities: tuple[PatchCapability, ...]
    await_receipts: tuple["AwaitReceipt", ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class TargetFactoryConformanceRunner:
    """Run one capability conformance corpus across every target profile."""

    profiles: tuple[TargetConformanceProfile, ...]

    def __post_init__(self) -> None:
        """Reject a profile corpus that omits or repeats a target context."""
        assert tuple(profile.kind for profile in self.profiles) == tuple(
            TargetProfileKind
        )

    async def run(
        self,
        factory: TargetConformanceFactory,
    ) -> tuple[TargetConformanceResult, ...]:
        """Return exact capability results without calling target mutations."""
        results: list[TargetConformanceResult] = []
        for profile in self.profiles:
            create_receipt = await _target_await(
                AwaitBoundary.TARGET_FACTORY_CREATE
            )
            target = await factory.create(profile)
            if not profile.capable:
                assert target is None
                results.append(
                    TargetConformanceResult(
                        kind=profile.kind,
                        capable=False,
                        capabilities=(),
                        await_receipts=(create_receipt,),
                    )
                )
                continue
            assert target is not None
            negotiate_receipt = await _target_await(
                AwaitBoundary.TARGET_FACTORY_NEGOTIATE
            )
            capabilities, _ = await target.negotiate_capabilities()
            assert set(profile.required_capabilities).issubset(capabilities)
            results.append(
                TargetConformanceResult(
                    kind=profile.kind,
                    capable=True,
                    capabilities=capabilities,
                    await_receipts=(create_receipt, negotiate_receipt),
                )
            )
        return tuple(results)


_CRASH_EXIT_CODE = 17
_CHILD_CRASH_PROGRAM = """import os
import sys

sys.stdout.buffer.write(b'barrier-ready\\n')
sys.stdout.buffer.flush()
command = sys.stdin.buffer.readline()
if command == b'crash\\n':
    os._exit(17)
if command == b'release\\n':
    sys.stdout.buffer.write(b'released\\n')
    sys.stdout.buffer.flush()
    raise SystemExit(0)
raise SystemExit(2)
"""


@dataclass(frozen=True, kw_only=True, slots=True)
class ChildProcessCrashReceipt:
    """Store exact IPC-barrier crash completion facts for one child process."""

    barrier: str
    exit_code: int
    stderr: bytes


@dataclass(kw_only=True, slots=True)
class ChildProcessCrashHarness:
    """Control one crashing child process through explicit pipe barriers."""

    process: Popen[bytes]

    @classmethod
    def start(cls) -> "ChildProcessCrashHarness":
        """Start one child and wait for its explicit IPC barrier message."""
        process = Popen(
            (executable, "-c", _CHILD_CRASH_PROGRAM),
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
        )
        harness = cls(process=process)
        assert harness._receive() == b"barrier-ready\n"
        return harness

    def trigger_crash(self) -> ChildProcessCrashReceipt:
        """Request an immediate crash through the reached IPC barrier."""
        self._send(b"crash\n")
        exit_code = self.process.wait()
        receipt = ChildProcessCrashReceipt(
            barrier="barrier-ready",
            exit_code=exit_code,
            stderr=self._read_stderr(),
        )
        self._close_pipes()
        assert receipt.exit_code == _CRASH_EXIT_CODE
        return receipt

    def release(self) -> ChildProcessCrashReceipt:
        """Release one child cleanly through the same explicit IPC barrier."""
        self._send(b"release\n")
        assert self._receive() == b"released\n"
        exit_code = self.process.wait()
        receipt = ChildProcessCrashReceipt(
            barrier="barrier-ready",
            exit_code=exit_code,
            stderr=self._read_stderr(),
        )
        self._close_pipes()
        assert receipt.exit_code == 0
        return receipt

    def _send(self, command: bytes) -> None:
        """Send one deterministic control message through child stdin."""
        assert self.process.stdin is not None
        self.process.stdin.write(command)
        self.process.stdin.flush()
        self.process.stdin.close()

    def _receive(self) -> bytes:
        """Receive one exact control message through the child stdout pipe."""
        assert self.process.stdout is not None
        message = self.process.stdout.readline()
        assert message
        return message

    def _read_stderr(self) -> bytes:
        """Read child stderr only after its process has exited."""
        assert self.process.stderr is not None
        return self.process.stderr.read()

    def _close_pipes(self) -> None:
        """Close the completed child pipes without sending a process signal."""
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self.process.stdout.close()
        self.process.stderr.close()


class ReviewerSeverity(str, Enum):
    """Name the closed severity levels for phase-evidence review findings."""

    P0 = "p0"
    P1 = "p1"
    P2 = "p2"
    P3 = "p3"
    P4 = "p4"


class ReviewerDisposition(str, Enum):
    """Name the closed dispositions for phase-evidence review findings."""

    FIXED = "fixed"
    ACCEPTED = "accepted"
    OPEN = "open"


class PhaseEvidenceStatus(str, Enum):
    """Name the lifecycle status of one immutable phase-evidence record."""

    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


@dataclass(frozen=True, kw_only=True, slots=True)
class ReviewerFinding:
    """Store one review finding and its explicit severity disposition."""

    identifier: str
    severity: ReviewerSeverity
    disposition: ReviewerDisposition
    rationale: str

    def __post_init__(self) -> None:
        """Reject an unnamed finding without a disposition rationale."""
        assert self.identifier and self.rationale


@dataclass(frozen=True, kw_only=True, slots=True)
class PhaseCommandEvidence:
    """Store one command and exact exit code in a phase-evidence record."""

    command: str
    exit_code: int

    def __post_init__(self) -> None:
        """Reject an empty command evidence record."""
        assert self.command


@dataclass(frozen=True, kw_only=True, slots=True)
class ArtifactDigest:
    """Store one named lower-case SHA-256 digest in phase evidence."""

    name: str
    sha256: PatchArtifactDigest

    def __post_init__(self) -> None:
        """Reject a non-canonical named SHA-256 artifact digest."""
        assert self.name and len(self.sha256) == 64
        assert all(
            character in "0123456789abcdef" for character in self.sha256
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class PhaseEvidence:
    """Store immutable commands, findings, nodes, and artifacts per phase."""

    phase: int
    status: PhaseEvidenceStatus
    active_node_ids: tuple[str, ...]
    commands: tuple[PhaseCommandEvidence, ...]
    artifact_digests: tuple[ArtifactDigest, ...]
    reviewer_findings: tuple[ReviewerFinding, ...]

    def __post_init__(self) -> None:
        """Reject an incomplete or non-deterministic phase-evidence record."""
        assert self.phase >= 0 and self.active_node_ids and self.commands
        assert tuple(sorted(self.active_node_ids)) == self.active_node_ids
        assert len(set(self.active_node_ids)) == len(self.active_node_ids)
        assert tuple(
            sorted(item.name for item in self.artifact_digests)
        ) == tuple(item.name for item in self.artifact_digests)
        assert len({item.name for item in self.artifact_digests}) == len(
            self.artifact_digests
        )
        assert tuple(
            sorted(item.identifier for item in self.reviewer_findings)
        ) == tuple(item.identifier for item in self.reviewer_findings)
        assert len(
            {item.identifier for item in self.reviewer_findings}
        ) == len(self.reviewer_findings)
        if self.status is PhaseEvidenceStatus.COMPLETE:
            assert not any(
                finding.disposition is ReviewerDisposition.OPEN
                and finding.severity
                in {
                    ReviewerSeverity.P0,
                    ReviewerSeverity.P1,
                    ReviewerSeverity.P2,
                }
                for finding in self.reviewer_findings
            )


@dataclass(frozen=True, kw_only=True, slots=True)
class PhaseEvidenceSigningMetadata:
    """Store the signer and canonical digest metadata for phase evidence."""

    algorithm: str
    signer_id: PatchObserverId
    signed_digest: PatchArtifactDigest

    def __post_init__(self) -> None:
        """Reject unsupported signing metadata or malformed signed digests."""
        assert self.algorithm == "sha256"
        ArtifactDigest(name="signed", sha256=self.signed_digest)


@dataclass(frozen=True, kw_only=True, slots=True)
class SealedPhaseEvidence:
    """Store immutable phase evidence together with its signing metadata."""

    evidence: PhaseEvidence
    canonical_digest: PatchArtifactDigest
    signing: PhaseEvidenceSigningMetadata

    def __post_init__(self) -> None:
        """Keep the evidence digest and signing metadata bound exactly."""
        assert self.canonical_digest == self.signing.signed_digest


class PhaseEvidenceCodec:
    """Encode, seal, and verify typed phase evidence deterministically."""

    @staticmethod
    def canonical_bytes(evidence: PhaseEvidence) -> bytes:
        """Return a canonical byte encoding without dynamic mapping input."""
        payload = b"phase-evidence-v1"
        payload += _encoded_text(str(evidence.phase))
        payload += _encoded_text(evidence.status.value)
        payload += b"nodes" + b"".join(
            _encoded_text(node_id) for node_id in evidence.active_node_ids
        )
        payload += b"commands" + b"".join(
            _encoded_text(command.command)
            + _encoded_text(str(command.exit_code))
            for command in evidence.commands
        )
        payload += b"artifacts" + b"".join(
            _encoded_text(artifact.name) + _encoded_text(artifact.sha256)
            for artifact in evidence.artifact_digests
        )
        payload += b"findings" + b"".join(
            _encoded_text(finding.identifier)
            + _encoded_text(finding.severity.value)
            + _encoded_text(finding.disposition.value)
            + _encoded_text(finding.rationale)
            for finding in evidence.reviewer_findings
        )
        return payload

    @classmethod
    def seal(
        cls,
        evidence: PhaseEvidence,
        signer_id: PatchObserverId,
    ) -> SealedPhaseEvidence:
        """Return immutable evidence with canonical digest signing metadata."""
        digest = PatchArtifactDigest(
            sha256(cls.canonical_bytes(evidence)).hexdigest()
        )
        return SealedPhaseEvidence(
            evidence=evidence,
            canonical_digest=digest,
            signing=PhaseEvidenceSigningMetadata(
                algorithm="sha256",
                signer_id=signer_id,
                signed_digest=digest,
            ),
        )

    @classmethod
    def verify(cls, sealed: SealedPhaseEvidence) -> bool:
        """Return whether one sealed record keeps its digest binding."""
        digest = PatchArtifactDigest(
            sha256(cls.canonical_bytes(sealed.evidence)).hexdigest()
        )
        return (
            sealed.canonical_digest == digest
            and sealed.signing.signed_digest == digest
            and sealed.signing.algorithm == "sha256"
        )


class ResourceOwner(str, Enum):
    """Name each resource whose depth is checked at an await boundary."""

    TRANSACTION = "transaction"
    COORDINATOR_LEASE = "coordinator_lease"
    TARGET_HANDLE = "target_handle"
    TARGET_WORKER = "target_worker"
    STAGING_RESOURCE = "staging_resource"
    APPROVAL_WAIT = "approval_wait"


class AwaitBoundary(str, Enum):
    """Name the closed await boundaries covered by the resource matrix."""

    STORE_CONNECTION = "store_connection"
    FAULT_WAIT = "fault_wait"
    TARGET_NEGOTIATION = "target_negotiation"
    TARGET_INSPECTION = "target_inspection"
    TARGET_PRECONDITION = "target_precondition"
    TARGET_HANDLE_OPEN = "target_handle_open"
    TARGET_HANDLE_CLOSE = "target_handle_close"
    TARGET_LOCK_ACQUIRE = "target_lock_acquire"
    TARGET_LOCK_RELEASE = "target_lock_release"
    TARGET_STAGE = "target_stage"
    TARGET_CLEANUP = "target_cleanup"
    TARGET_NAMESPACE_MUTATION = "target_namespace_mutation"
    TARGET_COMMIT = "target_commit"
    TARGET_VERIFICATION = "target_verification"
    APPROVAL_DECISION = "approval_decision"
    APPROVAL_CONSUME = "approval_consume"
    APPROVAL_CONCURRENT_CONSUME = "approval_concurrent_consume"
    TARGET_FACTORY_CREATE = "target_factory_create"
    TARGET_FACTORY_NEGOTIATE = "target_factory_negotiate"
    PUBLICATION = "publication"


class AllowedAwaitViolation(RuntimeError):
    """Report the exact owner depths forbidden at one await boundary."""


@dataclass(frozen=True, kw_only=True, slots=True)
class ResourceDepths:
    """Store immutable depths for resources tracked by the await matrix."""

    transaction: int = 0
    coordinator_lease: int = 0
    target_handle: int = 0
    target_worker: int = 0
    staging_resource: int = 0
    approval_wait: int = 0

    def __post_init__(self) -> None:
        """Reject a resource-depth vector with a negative owner depth."""
        assert all(depth >= 0 for depth in self.values())

    def values(self) -> tuple[int, ...]:
        """Return resource depths in the frozen owner order."""
        return (
            self.transaction,
            self.coordinator_lease,
            self.target_handle,
            self.target_worker,
            self.staging_resource,
            self.approval_wait,
        )

    def acquire(self, owner: ResourceOwner) -> "ResourceDepths":
        """Return depths after acquiring one named resource owner."""
        return self._change(owner, 1)

    def release(self, owner: ResourceOwner) -> "ResourceDepths":
        """Return depths after releasing one named resource owner."""
        return self._change(owner, -1)

    def diagnostic(self) -> str:
        """Return the exact nonzero owner-depth diagnostic string."""
        return (
            ",".join(
                f"{owner.value}:{depth}"
                for owner, depth in zip(
                    ResourceOwner, self.values(), strict=True
                )
                if depth
            )
            or "none"
        )

    def _change(self, owner: ResourceOwner, delta: int) -> "ResourceDepths":
        """Return depths after a checked immutable owner transition."""
        match owner:
            case ResourceOwner.TRANSACTION:
                next_depth = self.transaction + delta
                assert next_depth >= 0
                return ResourceDepths(
                    transaction=next_depth,
                    coordinator_lease=self.coordinator_lease,
                    target_handle=self.target_handle,
                    target_worker=self.target_worker,
                    staging_resource=self.staging_resource,
                    approval_wait=self.approval_wait,
                )
            case ResourceOwner.COORDINATOR_LEASE:
                next_depth = self.coordinator_lease + delta
                assert next_depth >= 0
                return ResourceDepths(
                    transaction=self.transaction,
                    coordinator_lease=next_depth,
                    target_handle=self.target_handle,
                    target_worker=self.target_worker,
                    staging_resource=self.staging_resource,
                    approval_wait=self.approval_wait,
                )
            case ResourceOwner.TARGET_HANDLE:
                next_depth = self.target_handle + delta
                assert next_depth >= 0
                return ResourceDepths(
                    transaction=self.transaction,
                    coordinator_lease=self.coordinator_lease,
                    target_handle=next_depth,
                    target_worker=self.target_worker,
                    staging_resource=self.staging_resource,
                    approval_wait=self.approval_wait,
                )
            case ResourceOwner.TARGET_WORKER:
                next_depth = self.target_worker + delta
                assert next_depth >= 0
                return ResourceDepths(
                    transaction=self.transaction,
                    coordinator_lease=self.coordinator_lease,
                    target_handle=self.target_handle,
                    target_worker=next_depth,
                    staging_resource=self.staging_resource,
                    approval_wait=self.approval_wait,
                )
            case ResourceOwner.STAGING_RESOURCE:
                next_depth = self.staging_resource + delta
                assert next_depth >= 0
                return ResourceDepths(
                    transaction=self.transaction,
                    coordinator_lease=self.coordinator_lease,
                    target_handle=self.target_handle,
                    target_worker=self.target_worker,
                    staging_resource=next_depth,
                    approval_wait=self.approval_wait,
                )
            case ResourceOwner.APPROVAL_WAIT:
                next_depth = self.approval_wait + delta
                assert next_depth >= 0
                return ResourceDepths(
                    transaction=self.transaction,
                    coordinator_lease=self.coordinator_lease,
                    target_handle=self.target_handle,
                    target_worker=self.target_worker,
                    staging_resource=self.staging_resource,
                    approval_wait=next_depth,
                )


def _allowed_await_depths(boundary: AwaitBoundary) -> ResourceDepths:
    """Return the only permitted resource-depth vector for one await point."""
    match boundary:
        case AwaitBoundary.STORE_CONNECTION:
            return ResourceDepths(transaction=1)
        case (
            AwaitBoundary.FAULT_WAIT
            | AwaitBoundary.TARGET_NEGOTIATION
            | AwaitBoundary.TARGET_INSPECTION
            | AwaitBoundary.TARGET_PRECONDITION
            | AwaitBoundary.TARGET_HANDLE_OPEN
            | AwaitBoundary.TARGET_NAMESPACE_MUTATION
            | AwaitBoundary.TARGET_FACTORY_CREATE
            | AwaitBoundary.TARGET_FACTORY_NEGOTIATE
            | AwaitBoundary.PUBLICATION
        ):
            return ResourceDepths()
        case AwaitBoundary.TARGET_HANDLE_CLOSE:
            return ResourceDepths(target_handle=1)
        case AwaitBoundary.TARGET_LOCK_ACQUIRE:
            return ResourceDepths(target_handle=1)
        case AwaitBoundary.TARGET_LOCK_RELEASE:
            return ResourceDepths(coordinator_lease=1, target_handle=1)
        case AwaitBoundary.TARGET_STAGE:
            return ResourceDepths(coordinator_lease=1, target_handle=1)
        case AwaitBoundary.TARGET_CLEANUP | AwaitBoundary.TARGET_VERIFICATION:
            return ResourceDepths(
                coordinator_lease=1,
                target_handle=1,
                staging_resource=1,
            )
        case AwaitBoundary.TARGET_COMMIT:
            return ResourceDepths(
                coordinator_lease=1,
                target_handle=1,
                target_worker=1,
                staging_resource=1,
            )
        case (
            AwaitBoundary.APPROVAL_DECISION
            | AwaitBoundary.APPROVAL_CONSUME
            | AwaitBoundary.APPROVAL_CONCURRENT_CONSUME
        ):
            return ResourceDepths(approval_wait=1)


@dataclass(frozen=True, kw_only=True, slots=True)
class AwaitReceipt:
    """Store one allowed await boundary and its exact resource depths."""

    boundary: AwaitBoundary
    depths: ResourceDepths


@dataclass(frozen=True, kw_only=True, slots=True)
class ResourceDepthSentinel:
    """Track depths and reject every unlisted await state."""

    depths: ResourceDepths = ResourceDepths()
    receipts: tuple[AwaitReceipt, ...] = ()

    def acquire(self, owner: ResourceOwner) -> "ResourceDepthSentinel":
        """Return a sentinel with one acquired owner depth."""
        return ResourceDepthSentinel(
            depths=self.depths.acquire(owner),
            receipts=self.receipts,
        )

    def release(self, owner: ResourceOwner) -> "ResourceDepthSentinel":
        """Return a sentinel with one released owner depth."""
        return ResourceDepthSentinel(
            depths=self.depths.release(owner),
            receipts=self.receipts,
        )

    async def at_await(
        self,
        boundary: AwaitBoundary,
    ) -> "ResourceDepthSentinel":
        """Record an allowed await or raise its exact depth diagnostic."""
        allowed = _allowed_await_depths(boundary)
        if self.depths != allowed:
            raise AllowedAwaitViolation(
                f"boundary={boundary.value} owners={self.depths.diagnostic()}"
            )
        return ResourceDepthSentinel(
            depths=self.depths,
            receipts=(
                *self.receipts,
                AwaitReceipt(boundary=boundary, depths=self.depths),
            ),
        )


async def _target_await(boundary: AwaitBoundary) -> AwaitReceipt:
    """Record one target await through the closed resource-depth sentinel."""
    sentinel = ResourceDepthSentinel()
    if boundary is AwaitBoundary.TARGET_COMMIT:
        for owner in (
            ResourceOwner.COORDINATOR_LEASE,
            ResourceOwner.TARGET_HANDLE,
            ResourceOwner.TARGET_WORKER,
            ResourceOwner.STAGING_RESOURCE,
        ):
            sentinel = sentinel.acquire(owner)
    checked = await sentinel.at_await(boundary)
    return checked.receipts[-1]


async def _approval_await(boundary: AwaitBoundary) -> AwaitReceipt:
    """Record one broker await through the closed approval-depth sentinel."""
    sentinel = ResourceDepthSentinel().acquire(ResourceOwner.APPROVAL_WAIT)
    checked = await sentinel.at_await(boundary)
    return checked.receipts[-1]
