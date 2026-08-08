"""Define immutable, dormant mutation-domain values and truth derivation."""

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from re import fullmatch
from secrets import token_hex
from typing import Self


class PatchDomainError(ValueError):
    """Report one expected domain outcome without exposing protected values."""


class PatchValidationError(PatchDomainError):
    """Report invalid untrusted boundary data."""


class PatchInvariantError(RuntimeError):
    """Report a programmer violation of an immutable domain invariant."""


@dataclass(frozen=True, slots=True)
class _PatchIdentifier:
    """Store one validated opaque identity with a fixed type-local prefix."""

    value: str

    _prefix: str = field(init=False, repr=False, default="")

    def __post_init__(self) -> None:
        """Validate a bounded, non-content-derived opaque identifier."""
        prefix = type(self)._identifier_prefix()
        if fullmatch(
            r"[a-z][a-z0-9_]{1,31}_[a-f0-9]{16,48}", self.value
        ) is None or not self.value.startswith(prefix):
            raise PatchValidationError(
                "identifier has an invalid prefix or body"
            )

    @classmethod
    def _identifier_prefix(cls) -> str:
        """Return the stable prefix assigned to one identity class."""
        raise NotImplementedError

    @classmethod
    def new(cls) -> Self:
        """Create an unpredictable identity independent of content."""
        return cls(cls._identifier_prefix() + token_hex(16))


@dataclass(frozen=True, slots=True)
class PatchRequestId(_PatchIdentifier):
    """Identify one trusted mutation request."""

    @classmethod
    def _identifier_prefix(cls) -> str:
        """Return the request identity prefix."""
        return "request_"


@dataclass(frozen=True, slots=True)
class PatchExecutionId(_PatchIdentifier):
    """Identify one execution binding."""

    @classmethod
    def _identifier_prefix(cls) -> str:
        """Return the execution identity prefix."""
        return "execution_"


@dataclass(frozen=True, slots=True)
class PatchPlanId(_PatchIdentifier):
    """Identify one immutable mutation plan."""

    @classmethod
    def _identifier_prefix(cls) -> str:
        """Return the plan identity prefix."""
        return "plan_"


@dataclass(frozen=True, slots=True)
class PatchOperationId(_PatchIdentifier):
    """Identify one canonical operation declaration."""

    @classmethod
    def _identifier_prefix(cls) -> str:
        """Return the operation identity prefix."""
        return "operation_"


@dataclass(frozen=True, slots=True)
class PatchLineageId(_PatchIdentifier):
    """Identify one canonical file lineage."""

    @classmethod
    def _identifier_prefix(cls) -> str:
        """Return the lineage identity prefix."""
        return "lineage_"


@dataclass(frozen=True, slots=True)
class PatchStepId(_PatchIdentifier):
    """Identify one requested-effect commit step."""

    @classmethod
    def _identifier_prefix(cls) -> str:
        """Return the step identity prefix."""
        return "step_"


@dataclass(frozen=True, slots=True)
class PatchContextId(_PatchIdentifier):
    """Identify one trusted execution context."""

    @classmethod
    def _identifier_prefix(cls) -> str:
        """Return the context identity prefix."""
        return "context_"


@dataclass(frozen=True, slots=True)
class PatchWorkspaceId(_PatchIdentifier):
    """Identify one selected workspace."""

    @classmethod
    def _identifier_prefix(cls) -> str:
        """Return the workspace identity prefix."""
        return "workspace_"


@dataclass(frozen=True, slots=True)
class PatchDomainId(_PatchIdentifier):
    """Identify one coordinated backing resource."""

    @classmethod
    def _identifier_prefix(cls) -> str:
        """Return the coordination-domain identity prefix."""
        return "domain_"


@dataclass(frozen=True, slots=True)
class PatchTargetId(_PatchIdentifier):
    """Identify one narrow mutation target."""

    @classmethod
    def _identifier_prefix(cls) -> str:
        """Return the target identity prefix."""
        return "target_"


@dataclass(frozen=True, slots=True)
class PatchProtocolId(_PatchIdentifier):
    """Identify one target protocol version."""

    @classmethod
    def _identifier_prefix(cls) -> str:
        """Return the protocol identity prefix."""
        return "protocol_"


@dataclass(frozen=True, slots=True)
class PatchGrantId(_PatchIdentifier):
    """Identify one trusted approval-grant record."""

    @classmethod
    def _identifier_prefix(cls) -> str:
        """Return the grant identity prefix."""
        return "grant_"


@dataclass(frozen=True, slots=True)
class PatchApprovalId(_PatchIdentifier):
    """Identify one approval decision."""

    @classmethod
    def _identifier_prefix(cls) -> str:
        """Return the approval identity prefix."""
        return "approval_"


@dataclass(frozen=True, slots=True)
class PatchEventId(_PatchIdentifier):
    """Identify one lifecycle event."""

    @classmethod
    def _identifier_prefix(cls) -> str:
        """Return the event identity prefix."""
        return "event_"


@dataclass(frozen=True, slots=True)
class PatchPendingOperationId(_PatchIdentifier):
    """Identify one nonterminal settlement operation."""

    @classmethod
    def _identifier_prefix(cls) -> str:
        """Return the pending-operation identity prefix."""
        return "pending_"


@dataclass(frozen=True, slots=True)
class PatchObserverId(_PatchIdentifier):
    """Identify one random observer correlation."""

    @classmethod
    def _identifier_prefix(cls) -> str:
        """Return the observer identity prefix."""
        return "observer_"


@dataclass(frozen=True, slots=True)
class PatchObserverCorrelationId(_PatchIdentifier):
    """Identify one observer-local correlation."""

    @classmethod
    def _identifier_prefix(cls) -> str:
        """Return the observer-correlation identity prefix."""
        return "correlation_"


@dataclass(frozen=True, slots=True)
class LogicalPath:
    """Store one canonical workspace-relative logical path."""

    value: str

    def __post_init__(self) -> None:
        """Reject absolute, ambiguous, and platform-specific paths."""
        parts = self.value.split("/")
        if (
            not self.value
            or len(self.value) > 1024
            or self.value.startswith("/")
            or "\\" in self.value
            or any(part in {"", ".", ".."} for part in parts)
            or any(len(part) > 255 for part in parts)
        ):
            raise PatchValidationError("logical path is invalid")


@dataclass(frozen=True, order=True, slots=True)
class ByteSize:
    """Store one bounded nonnegative byte size."""

    value: int

    def __post_init__(self) -> None:
        """Reject sizes outside the portable domain range."""
        if type(self.value) is not int or not 0 <= self.value <= 2**63 - 1:
            raise PatchValidationError("byte size is invalid")


@dataclass(frozen=True, order=True, slots=True)
class SequenceNumber:
    """Store one monotonic nonnegative lifecycle sequence."""

    value: int

    def __post_init__(self) -> None:
        """Reject invalid sequence numbers."""
        if type(self.value) is not int or not 0 <= self.value <= 2**63 - 1:
            raise PatchValidationError("sequence number is invalid")


@dataclass(frozen=True, order=True, slots=True)
class DurationTicks:
    """Store one finite positive duration measured by a trusted clock."""

    value: int

    def __post_init__(self) -> None:
        """Reject nonpositive or unbounded duration values."""
        if type(self.value) is not int or not 1 <= self.value <= 2**31 - 1:
            raise PatchValidationError("duration is invalid")


@dataclass(frozen=True, order=True, slots=True)
class ExpiryTick:
    """Store one finite positive expiry point on a trusted clock."""

    value: int

    def __post_init__(self) -> None:
        """Reject invalid expiry values."""
        if type(self.value) is not int or not 1 <= self.value <= 2**63 - 1:
            raise PatchValidationError("expiry tick is invalid")


@dataclass(frozen=True, slots=True)
class AlgorithmDigest:
    """Store one algorithm-qualified fixed-width integrity digest."""

    algorithm: str
    value: str

    def __post_init__(self) -> None:
        """Restrict the initial domain to lowercase SHA-256 evidence."""
        if (
            self.algorithm != "sha256"
            or fullmatch(r"[0-9a-f]{64}", self.value) is None
        ):
            raise PatchValidationError("digest is invalid")

    @classmethod
    def from_bytes(cls, value: bytes) -> "AlgorithmDigest":
        """Return the SHA-256 digest of a bounded internal byte value."""
        return cls(algorithm="sha256", value=sha256(value).hexdigest())


@dataclass(frozen=True, slots=True)
class FileMode:
    """Store a portable POSIX file mode without ownership metadata."""

    value: int

    def __post_init__(self) -> None:
        """Reject nonportable mode bits."""
        if type(self.value) is not int or not 0 <= self.value <= 0o777:
            raise PatchValidationError("file mode is invalid")


class OperationType(str, Enum):
    """Name one closed semantic request operation."""

    EDIT = "edit"
    APPLY = "apply"


class Capability(str, Enum):
    """Name one independently authorized effect or inspection capability."""

    UPDATE = "update"
    CREATE = "create"
    DELETE = "delete"
    MOVE = "move"
    READ_FOR_MUTATION = "read_for_mutation"
    OBSERVE_MUTATION_PRECONDITIONS = "observe_mutation_preconditions"
    UPDATE_EXECUTABLE = "update_executable"


class Disclosure(str, Enum):
    """Name one separately controlled disclosure capability."""

    REVIEW_DIFF = "review_diff"
    HASHES_AND_SIZES = "hashes_and_sizes"
    MATCH_DETAILS = "match_details"
    LOGICAL_PATHS = "logical_paths"


class ApprovalMode(str, Enum):
    """Name the only policy-selected approval modes."""

    DENY = "deny"
    REQUIRE_REVIEW = "require_review"
    PREAUTHORIZED = "preauthorized"


class ContextKind(str, Enum):
    """Name the trusted mutation context kinds."""

    LOCAL = "local"
    SANDBOX = "sandbox"
    CONTAINER = "container"


class MatchStrategy(str, Enum):
    """Name lossless matching strategies allowed by the domain."""

    EXACT_BYTES = "exact_bytes"
    REPRESENTATION_COMPATIBLE = "representation_compatible"


class AtomicityClass(str, Enum):
    """Name the declared commit atomicity class."""

    SINGLE_STEP = "single_step"
    PER_LINEAGE = "per_lineage"
    BEST_EFFORT = "best_effort"


class CommitStepState(str, Enum):
    """Name the journal truth of one requested-effect step."""

    PLANNED = "planned"
    NOT_COMMITTED = "not_committed"
    COMMITTED = "committed"
    UNKNOWN = "unknown"


class LineageState(str, Enum):
    """Name the requested-effect truth for one terminal lineage."""

    NOT_COMMITTED = "not_committed"
    COMMITTED = "committed"
    PARTIALLY_COMMITTED = "partially_committed"
    INDETERMINATE = "indeterminate"


class MutationState(str, Enum):
    """Name the journal-derived truth of all requested effects."""

    NOT_COMMITTED = "not_committed"
    COMMITTED = "committed"
    PARTIALLY_COMMITTED = "partially_committed"
    INDETERMINATE = "indeterminate"


class ArtifactState(str, Enum):
    """Name the independent state of target-private staging artifacts."""

    ABSENT = "absent"
    STAGED = "staged"
    CLEANED = "cleaned"
    LEAKED = "leaked"
    UNKNOWN = "unknown"


class PostconditionState(str, Enum):
    """Name the current observed state of a requested terminal effect."""

    ESTABLISHED = "established"
    SUPERSEDED = "superseded"
    UNKNOWN = "unknown"


class RequestedEffectOccurrence(str, Enum):
    """Name whether a requested effect occurred independently of artifacts."""

    FALSE = "false"
    TRUE = "true"
    UNKNOWN = "unknown"


class WorkspaceChange(str, Enum):
    """Name whether the currently observed workspace differs from baseline."""

    UNCHANGED = "unchanged"
    CHANGED = "changed"
    UNKNOWN = "unknown"


class LifecyclePhase(str, Enum):
    """Name one state in the exact logical request lifecycle."""

    RECEIVED = "received"
    PARSED = "parsed"
    SCOPE_BOUND = "scope_bound"
    PREFLIGHT_AUTHORIZED = "preflight_authorized"
    PLANNED = "planned"
    APPROVAL_REQUIRED = "approval_required"
    APPROVED = "approved"
    COMMIT_READY = "commit_ready"
    COMMIT_STARTED = "commit_started"
    SETTLEMENT_PENDING = "settlement_pending"
    SETTLED = "settled"
    REQUEST_COMPLETED = "request_completed"


class PatchStatus(str, Enum):
    """Name one closed terminal status independent of mutation truth."""

    REJECTED = "rejected"
    DENIED = "denied"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_UNAVAILABLE = "approval_unavailable"
    STALE = "stale"
    CANCELLED = "cancelled"
    COMMIT_FAILED = "commit_failed"
    COMMITTED = "committed"
    PARTIAL = "partial"
    INDETERMINATE = "indeterminate"


class ErrorStage(str, Enum):
    """Name the stage responsible for a stable error code."""

    INPUT = "input"
    SCOPE = "scope"
    PREFLIGHT = "preflight"
    PLANNING = "planning"
    APPROVAL = "approval"
    REVALIDATION = "revalidation"
    COMMIT = "commit"
    SETTLEMENT = "settlement"


class PatchErrorCode(str, Enum):
    """Name stable public and privileged error codes without free-form text."""

    INVALID_REQUEST = "patch.invalid_request"
    INVALID_PATCH = "patch.invalid_patch"
    UNSUPPORTED_OPERATION = "patch.unsupported_operation"
    CONFLICTING_OPERATIONS = "patch.conflicting_operations"
    NO_EFFECT = "patch.no_effect"
    LIMIT_EXCEEDED = "patch.limit_exceeded"
    CONTEXT_UNAVAILABLE = "patch.context_unavailable"
    BACKEND_UNAVAILABLE = "patch.backend_unavailable"
    CAPABILITY_UNAVAILABLE = "patch.capability_unavailable"
    CAPABILITY_REQUIRED = "patch.capability_required"
    PRECONDITION_OBSERVATION_REQUIRED = (
        "patch.precondition_observation_required"
    )
    PATH_DENIED = "patch.path_denied"
    TRAVERSAL_DENIED = "patch.traversal_denied"
    LINK_DENIED = "patch.link_denied"
    ALIAS_DENIED = "patch.alias_denied"
    MOUNT_DENIED = "patch.mount_denied"
    SPECIAL_FILE_DENIED = "patch.special_file_denied"
    PARENT_MISSING = "patch.parent_missing"
    SOURCE_MISSING = "patch.source_missing"
    DESTINATION_EXISTS = "patch.destination_exists"
    METADATA_UNSUPPORTED = "patch.metadata_unsupported"
    UNSUPPORTED_CONTENT = "patch.unsupported_content"
    ENCODING_UNSUPPORTED = "patch.encoding_unsupported"
    REPRESENTATION_UNSUPPORTED = "patch.representation_unsupported"
    MATCH_NOT_FOUND = "patch.match_not_found"
    AMBIGUOUS_MATCH = "patch.ambiguous_match"
    OVERLAPPING_EDITS = "patch.overlapping_edits"
    APPROVAL_REQUIRED = "patch.approval_required"
    APPROVAL_DENIED = "patch.approval_denied"
    APPROVAL_UNAVAILABLE = "patch.approval_unavailable"
    APPROVAL_EXPIRED = "patch.approval_expired"
    APPROVAL_MISMATCH = "patch.approval_mismatch"
    STALE = "patch.stale"
    CANCELLED = "patch.cancelled"
    TIMEOUT = "patch.timeout"
    COMMIT_FAILED = "patch.commit_failed"
    PARTIAL_COMMIT = "patch.partial_commit"
    INDETERMINATE = "patch.indeterminate"
    VERIFICATION_FAILED = "patch.verification_failed"
    STAGING_ARTIFACT_LEAKED = "patch.staging_artifact_leaked"
    STAGING_ARTIFACT_UNKNOWN = "patch.staging_artifact_unknown"
    DIAGNOSTIC_FAILED = "patch.diagnostic_failed"


class Retryability(str, Enum):
    """Name the only retry classifications visible to the coordinator."""

    RETRYABLE_PRECOMMIT = "retryable_precommit"
    RETRANSMIT_ONLY = "retransmit_only"
    NOT_RETRYABLE = "not_retryable"


class Audience(str, Enum):
    """Name one audience that must receive an explicit projection."""

    PUBLIC = "public"
    MODEL = "model"
    APPROVER = "approver"
    AUDIT = "audit"
    OPERATOR = "operator"


@dataclass(frozen=True, slots=True, repr=False)
class _RedactedBytes:
    """Store protected bytes without admitting them to default rendering."""

    _value: bytes = field(repr=False)

    def __post_init__(self) -> None:
        """Copy immutable bytes at the trust boundary."""
        if not isinstance(self._value, bytes):
            raise PatchValidationError("protected value must be bytes")

    def __repr__(self) -> str:
        """Render a stable type-only redaction marker."""
        return f"{type(self).__name__}(<redacted>)"

    def __str__(self) -> str:
        """Return the stable redaction marker."""
        return "<redacted>"

    def digest(self) -> AlgorithmDigest:
        """Return integrity evidence without returning protected bytes."""
        return AlgorithmDigest.from_bytes(self._value)

    def size(self) -> ByteSize:
        """Return the protected byte count as a typed value."""
        return ByteSize(len(self._value))


@dataclass(frozen=True, slots=True, repr=False)
class PatchInput(_RedactedBytes):
    """Store raw patch input bytes."""


@dataclass(frozen=True, slots=True, repr=False)
class SourceBytes(_RedactedBytes):
    """Store private source bytes."""


@dataclass(frozen=True, slots=True, repr=False)
class ProposedBytes(_RedactedBytes):
    """Store private proposed bytes."""


@dataclass(frozen=True, slots=True, repr=False)
class DiffBytes(_RedactedBytes):
    """Store a complete privileged review diff."""


@dataclass(frozen=True, slots=True, repr=False)
class PatchFingerprint(_RedactedBytes):
    """Store the opaque content-bound plan fingerprint."""


@dataclass(frozen=True, slots=True, repr=False)
class GrantSecret(_RedactedBytes):
    """Store opaque grant material outside projections."""


@dataclass(frozen=True, slots=True, repr=False)
class PrivateStagingName(_RedactedBytes):
    """Store a target-private artifact name."""


@dataclass(frozen=True, slots=True, repr=False)
class PatchCredential(_RedactedBytes):
    """Store one protected credential value."""


@dataclass(frozen=True, slots=True)
class MetadataProfile:
    """Store protected representation metadata without mutable maps."""

    mode: FileMode
    has_utf8_bom: bool
    newline: str

    def __post_init__(self) -> None:
        """Restrict metadata to supported text representations."""
        if self.newline not in {"lf", "crlf"}:
            raise PatchValidationError("metadata newline is invalid")


@dataclass(frozen=True, slots=True)
class PatchLimits:
    """Store finite policy limits for the future effectful layers."""

    input_bytes: ByteSize
    path_count: ByteSize
    path_length: ByteSize
    file_count: ByteSize
    operation_count: ByteSize
    snapshot_bytes: ByteSize
    proposed_bytes: ByteSize
    review_diff_bytes: ByteSize
    planning_duration: DurationTicks
    approval_duration: DurationTicks
    commit_duration: DurationTicks

    def __post_init__(self) -> None:
        """Require every limit to be finite and nonzero."""
        if any(
            value.value == 0
            for value in (
                self.input_bytes,
                self.path_count,
                self.path_length,
                self.file_count,
                self.operation_count,
                self.snapshot_bytes,
                self.proposed_bytes,
                self.review_diff_bytes,
            )
        ):
            raise PatchValidationError("patch limits must be nonzero")


@dataclass(frozen=True, slots=True)
class PatchRequest:
    """Store a validated semantic request without a decoded grammar payload."""

    schema_version: int
    request_id: PatchRequestId
    execution_id: PatchExecutionId
    operation: OperationType
    input_bytes: PatchInput
    logical_paths: tuple[LogicalPath, ...]

    def __post_init__(self) -> None:
        """Require a closed initial request schema and unique paths."""
        if (
            self.schema_version != 1
            or type(self.logical_paths) is not tuple
            or not self.logical_paths
            or any(
                type(path) is not LogicalPath for path in self.logical_paths
            )
        ):
            raise PatchValidationError("patch request is invalid")
        if len(set(self.logical_paths)) != len(self.logical_paths):
            raise PatchValidationError("patch request paths are duplicated")


@dataclass(frozen=True, slots=True)
class MutationScope:
    """Store trusted context and authority selections for one request."""

    context_kind: ContextKind
    context_id: PatchContextId
    workspace_id: PatchWorkspaceId
    domain_id: PatchDomainId
    target_id: PatchTargetId
    protocol_id: PatchProtocolId
    capabilities: frozenset[Capability]
    disclosures: frozenset[Disclosure]
    limits: PatchLimits

    def __post_init__(self) -> None:
        """Reject mutable or untyped authority collections at the boundary."""
        if (
            type(self.capabilities) is not frozenset
            or type(self.disclosures) is not frozenset
            or not self.capabilities
            or any(type(item) is not Capability for item in self.capabilities)
            or any(type(item) is not Disclosure for item in self.disclosures)
        ):
            raise PatchValidationError("mutation scope authority is invalid")


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Store one immutable before-state observation."""

    path: LogicalPath
    present: bool
    size: ByteSize
    digest: AlgorithmDigest | None
    metadata: MetadataProfile | None

    def __post_init__(self) -> None:
        """Keep absent and present snapshot facts structurally consistent."""
        if self.present and (self.digest is None or self.metadata is None):
            raise PatchValidationError("present snapshot lacks representation")
        if not self.present and (
            self.size.value != 0
            or self.digest is not None
            or self.metadata is not None
        ):
            raise PatchValidationError("absent snapshot has observed content")


@dataclass(frozen=True, slots=True)
class VirtualFile:
    """Store one in-memory proposed file value."""

    path: LogicalPath
    bytes_value: ProposedBytes
    metadata: MetadataProfile


@dataclass(frozen=True, slots=True)
class CommitGraph:
    """Store the immutable irreversible requested-effect ordering."""

    steps: tuple[PatchStepId, ...]
    atomicity: AtomicityClass

    def __post_init__(self) -> None:
        """Require a nonempty unique graph of named steps."""
        if (
            type(self.steps) is not tuple
            or not self.steps
            or any(type(step) is not PatchStepId for step in self.steps)
            or len(set(self.steps)) != len(self.steps)
        ):
            raise PatchValidationError("commit graph is invalid")


@dataclass(frozen=True, slots=True)
class Lineage:
    """Store a canonical initial-to-final logical file lineage."""

    lineage_id: PatchLineageId
    source_path: LogicalPath | None
    destination_path: LogicalPath | None
    required_capabilities: frozenset[Capability]
    match_strategy: MatchStrategy | None
    commit_graph: CommitGraph

    def __post_init__(self) -> None:
        """Reject lineages with neither an initial nor final logical path."""
        if (
            (self.source_path is None and self.destination_path is None)
            or type(self.required_capabilities) is not frozenset
            or any(
                type(item) is not Capability
                for item in self.required_capabilities
            )
        ):
            raise PatchValidationError("lineage has no visible path")


@dataclass(frozen=True, slots=True)
class ReviewArtifact:
    """Store the complete private review representation of a sealed plan."""

    diff: DiffBytes
    digest: AlgorithmDigest
    size: ByteSize

    def __post_init__(self) -> None:
        """Bind declared integrity facts to the private diff value."""
        if self.digest != self.diff.digest() or self.size != self.diff.size():
            raise PatchInvariantError(
                "review artifact facts do not match bytes"
            )


@dataclass(frozen=True, slots=True)
class MutationPlan:
    """Store one immutable plan without any mutation capability."""

    plan_id: PatchPlanId
    request: PatchRequest
    scope: MutationScope
    lineages: tuple[Lineage, ...]
    review: ReviewArtifact
    fingerprint: PatchFingerprint

    def __post_init__(self) -> None:
        """Require a plan to own unique canonical terminal lineages."""
        if (
            type(self.lineages) is not tuple
            or not self.lineages
            or any(type(item) is not Lineage for item in self.lineages)
            or len({item.lineage_id for item in self.lineages})
            != len(self.lineages)
        ):
            raise PatchValidationError("mutation plan lineages are invalid")


@dataclass(frozen=True, slots=True)
class ApprovalGrant:
    """Store trusted opaque approval material bound to one plan identity."""

    grant_id: PatchGrantId
    approval_id: PatchApprovalId
    plan_id: PatchPlanId
    expiry: ExpiryTick
    secret: GrantSecret


@dataclass(frozen=True, slots=True)
class CommitStepJournal:
    """Store immutable observed truth for one requested-effect step."""

    step_id: PatchStepId
    state: CommitStepState


@dataclass(frozen=True, slots=True)
class LineageJournal:
    """Store one lineage's requested-effect and current postcondition facts."""

    lineage_id: PatchLineageId
    steps: tuple[CommitStepJournal, ...]
    postcondition: PostconditionState
    artifact_state: ArtifactState

    def __post_init__(self) -> None:
        """Require a lineage journal to retain its complete step vector."""
        if (
            type(self.steps) is not tuple
            or not self.steps
            or any(type(item) is not CommitStepJournal for item in self.steps)
            or len({item.step_id for item in self.steps}) != len(self.steps)
        ):
            raise PatchValidationError("lineage journal is invalid")


@dataclass(frozen=True, slots=True)
class CommitTruth:
    """Store orthogonal requested-effect, artifact, and workspace facts."""

    mutation_state: MutationState
    lineage_state: LineageState
    requested_effect_occurred: RequestedEffectOccurrence
    artifact_state: ArtifactState
    workspace_change: WorkspaceChange
    commit_set_exact: bool
    postcondition: PostconditionState

    def __post_init__(self) -> None:
        """Reject contradictory aggregate commit and occurrence facts."""
        if self.lineage_state.value != self.mutation_state.value:
            raise PatchValidationError("commit and lineage truth disagree")
        if self.commit_set_exact is not (
            self.mutation_state is not MutationState.INDETERMINATE
        ):
            raise PatchValidationError("commit exactness truth is invalid")
        if self.mutation_state is MutationState.NOT_COMMITTED:
            expected_occurrence = RequestedEffectOccurrence.FALSE
        elif self.mutation_state in {
            MutationState.COMMITTED,
            MutationState.PARTIALLY_COMMITTED,
        }:
            expected_occurrence = RequestedEffectOccurrence.TRUE
        else:
            expected_occurrence = self.requested_effect_occurred
            if expected_occurrence not in {
                RequestedEffectOccurrence.TRUE,
                RequestedEffectOccurrence.UNKNOWN,
            }:
                raise PatchValidationError(
                    "indeterminate occurrence is invalid"
                )
        if self.requested_effect_occurred is not expected_occurrence:
            raise PatchValidationError(
                "requested-effect occurrence is invalid"
            )
        if self.postcondition is not PostconditionState.UNKNOWN and (
            self.requested_effect_occurred
            is not RequestedEffectOccurrence.TRUE
        ):
            raise PatchValidationError(
                "postcondition contradicts commit truth"
            )
        if self.workspace_change is not _workspace_change(
            self.requested_effect_occurred, self.artifact_state
        ):
            raise PatchValidationError("workspace-change truth is invalid")


@dataclass(frozen=True, slots=True)
class PatchDiagnostic:
    """Store a stable internal diagnostic independent of mutation truth."""

    stage: ErrorStage
    code: PatchErrorCode
    retryability: Retryability


def coarsen_error_code(
    code: PatchErrorCode, audience: Audience
) -> PatchErrorCode:
    """Project protected source detail to a public-safe stable error code."""
    if audience in {Audience.PUBLIC, Audience.MODEL}:
        if code in {
            PatchErrorCode.SOURCE_MISSING,
            PatchErrorCode.DESTINATION_EXISTS,
            PatchErrorCode.LINK_DENIED,
            PatchErrorCode.ALIAS_DENIED,
            PatchErrorCode.MOUNT_DENIED,
            PatchErrorCode.SPECIAL_FILE_DENIED,
        }:
            return PatchErrorCode.PATH_DENIED
        if code in {
            PatchErrorCode.UNSUPPORTED_CONTENT,
            PatchErrorCode.ENCODING_UNSUPPORTED,
            PatchErrorCode.REPRESENTATION_UNSUPPORTED,
        }:
            return PatchErrorCode.INVALID_REQUEST
    return code


@dataclass(frozen=True, slots=True)
class PatchResult:
    """Store the sole terminal mutation result projection input."""

    schema_version: int
    request_id: PatchRequestId
    plan_id: PatchPlanId
    lifecycle: LifecyclePhase
    status: PatchStatus
    truth: CommitTruth
    diagnostic: PatchDiagnostic

    def __post_init__(self) -> None:
        """Require terminal lifecycle and status/truth combinations."""
        if (
            self.schema_version != 1
            or self.lifecycle is not LifecyclePhase.REQUEST_COMPLETED
        ):
            raise PatchValidationError("patch result is not terminal")
        _validate_status_truth(self.status, self.truth.mutation_state)


@dataclass(frozen=True, slots=True)
class PatchPending:
    """Store a public-safe nonterminal settlement envelope."""

    schema_version: int
    pending_operation_id: PatchPendingOperationId
    request_id: PatchRequestId
    correlation_id: PatchObserverCorrelationId
    lifecycle: LifecyclePhase

    def __post_init__(self) -> None:
        """Require a pending envelope to remain exactly nonterminal."""
        if (
            self.schema_version != 1
            or self.lifecycle is not LifecyclePhase.SETTLEMENT_PENDING
        ):
            raise PatchValidationError("pending envelope is not nonterminal")


PatchInvocationOutcome = PatchResult | PatchPending


@dataclass(frozen=True, slots=True)
class PatchLifecycleEvent:
    """Store one content-free canonical lifecycle event."""

    schema_version: int
    event_id: PatchEventId
    observer_id: PatchObserverId
    correlation_id: PatchObserverCorrelationId
    request_id: PatchRequestId
    sequence: SequenceNumber
    lifecycle: LifecyclePhase

    def __post_init__(self) -> None:
        """Require the fixed event schema version."""
        if self.schema_version != 1:
            raise PatchValidationError("lifecycle event version is invalid")


@dataclass(frozen=True, slots=True)
class PublicPendingProjection:
    """Store the only public-safe pending operation representation."""

    schema_version: int
    pending_operation_id: PatchPendingOperationId
    correlation_id: PatchObserverCorrelationId
    lifecycle: LifecyclePhase


@dataclass(frozen=True, slots=True)
class ProjectionInput:
    """Store explicit audience and canonical outcome before projection."""

    audience: Audience
    outcome: PatchInvocationOutcome


def project_pending(value: ProjectionInput) -> PublicPendingProjection:
    """Project a pending outcome only through an explicit audience boundary."""
    match value.outcome:
        case PatchPending() as pending:
            return PublicPendingProjection(
                schema_version=pending.schema_version,
                pending_operation_id=pending.pending_operation_id,
                correlation_id=pending.correlation_id,
                lifecycle=pending.lifecycle,
            )
        case PatchResult():
            raise PatchValidationError(
                "terminal result cannot project as pending"
            )


@dataclass(frozen=True, slots=True)
class LifecycleTransition:
    """Store one exact allowed transition in the closed lifecycle algebra."""

    current: LifecyclePhase
    next: LifecyclePhase


_TRANSITIONS = frozenset(
    (
        LifecycleTransition(LifecyclePhase.RECEIVED, LifecyclePhase.PARSED),
        LifecycleTransition(
            LifecyclePhase.RECEIVED, LifecyclePhase.REQUEST_COMPLETED
        ),
        LifecycleTransition(LifecyclePhase.PARSED, LifecyclePhase.SCOPE_BOUND),
        LifecycleTransition(
            LifecyclePhase.PARSED, LifecyclePhase.REQUEST_COMPLETED
        ),
        LifecycleTransition(
            LifecyclePhase.SCOPE_BOUND, LifecyclePhase.PREFLIGHT_AUTHORIZED
        ),
        LifecycleTransition(
            LifecyclePhase.SCOPE_BOUND, LifecyclePhase.REQUEST_COMPLETED
        ),
        LifecycleTransition(
            LifecyclePhase.PREFLIGHT_AUTHORIZED, LifecyclePhase.PLANNED
        ),
        LifecycleTransition(
            LifecyclePhase.PREFLIGHT_AUTHORIZED,
            LifecyclePhase.REQUEST_COMPLETED,
        ),
        LifecycleTransition(
            LifecyclePhase.PLANNED, LifecyclePhase.REQUEST_COMPLETED
        ),
        LifecycleTransition(
            LifecyclePhase.PLANNED, LifecyclePhase.APPROVAL_REQUIRED
        ),
        LifecycleTransition(LifecyclePhase.PLANNED, LifecyclePhase.APPROVED),
        LifecycleTransition(
            LifecyclePhase.APPROVAL_REQUIRED, LifecyclePhase.APPROVED
        ),
        LifecycleTransition(
            LifecyclePhase.APPROVAL_REQUIRED, LifecyclePhase.REQUEST_COMPLETED
        ),
        LifecycleTransition(
            LifecyclePhase.APPROVED, LifecyclePhase.COMMIT_READY
        ),
        LifecycleTransition(
            LifecyclePhase.APPROVED, LifecyclePhase.REQUEST_COMPLETED
        ),
        LifecycleTransition(
            LifecyclePhase.COMMIT_READY, LifecyclePhase.COMMIT_STARTED
        ),
        LifecycleTransition(
            LifecyclePhase.COMMIT_STARTED, LifecyclePhase.SETTLED
        ),
        LifecycleTransition(
            LifecyclePhase.COMMIT_STARTED, LifecyclePhase.SETTLEMENT_PENDING
        ),
        LifecycleTransition(
            LifecyclePhase.SETTLEMENT_PENDING, LifecyclePhase.SETTLED
        ),
        LifecycleTransition(
            LifecyclePhase.SETTLED, LifecyclePhase.REQUEST_COMPLETED
        ),
    )
)


def advance_lifecycle(
    current: LifecyclePhase, next: LifecyclePhase
) -> LifecyclePhase:
    """Validate and return one exact lifecycle transition."""
    if LifecycleTransition(current=current, next=next) not in _TRANSITIONS:
        raise PatchValidationError("lifecycle transition is invalid")
    return next


def derive_commit_truth(
    plan: MutationPlan, journals: tuple[LineageJournal, ...]
) -> CommitTruth:
    """Derive request-wide truth from a sealed plan and bound journals."""
    _validate_journal_binding(plan, journals)
    states = tuple(
        step.state for journal in journals for step in journal.steps
    )
    if CommitStepState.PLANNED in states:
        raise PatchValidationError("settled journal retains planned step")
    has_unknown = CommitStepState.UNKNOWN in states
    has_committed = CommitStepState.COMMITTED in states
    committed = sum(state is CommitStepState.COMMITTED for state in states)
    if has_unknown:
        mutation = MutationState.INDETERMINATE
        lineage = LineageState.INDETERMINATE
        occurrence = (
            RequestedEffectOccurrence.TRUE
            if has_committed
            else RequestedEffectOccurrence.UNKNOWN
        )
    elif committed == 0:
        mutation = MutationState.NOT_COMMITTED
        lineage = LineageState.NOT_COMMITTED
        occurrence = RequestedEffectOccurrence.FALSE
    elif committed == len(states):
        mutation = MutationState.COMMITTED
        lineage = LineageState.COMMITTED
        occurrence = RequestedEffectOccurrence.TRUE
    else:
        mutation = MutationState.PARTIALLY_COMMITTED
        lineage = LineageState.PARTIALLY_COMMITTED
        occurrence = RequestedEffectOccurrence.TRUE
    artifact = _aggregate_artifact_state(journals)
    postcondition = _aggregate_postcondition(journals, occurrence)
    workspace = _workspace_change(occurrence, artifact)
    return CommitTruth(
        mutation_state=mutation,
        lineage_state=lineage,
        requested_effect_occurred=occurrence,
        artifact_state=artifact,
        workspace_change=workspace,
        commit_set_exact=not has_unknown,
        postcondition=postcondition,
    )


def _validate_journal_binding(
    plan: MutationPlan, journals: tuple[LineageJournal, ...]
) -> None:
    """Require journals to exactly retain the sealed lineage step graphs."""
    if type(journals) is not tuple or len(journals) != len(plan.lineages):
        raise PatchValidationError("journals do not match the sealed plan")
    graphs = {
        lineage.lineage_id: lineage.commit_graph.steps
        for lineage in plan.lineages
    }
    observed = tuple(journal.lineage_id for journal in journals)
    if len(set(observed)) != len(observed) or set(observed) != set(graphs):
        raise PatchValidationError("journal lineages do not match the plan")
    for journal in journals:
        expected_steps = graphs[journal.lineage_id]
        observed_steps = tuple(step.step_id for step in journal.steps)
        if observed_steps != expected_steps:
            raise PatchValidationError("journal steps do not match the plan")


def _aggregate_artifact_state(
    journals: tuple[LineageJournal, ...],
) -> ArtifactState:
    """Aggregate independent staging facts without inferring effects."""
    artifacts = frozenset(journal.artifact_state for journal in journals)
    if ArtifactState.UNKNOWN in artifacts:
        return ArtifactState.UNKNOWN
    if ArtifactState.LEAKED in artifacts:
        return ArtifactState.LEAKED
    if ArtifactState.STAGED in artifacts:
        return ArtifactState.STAGED
    if ArtifactState.CLEANED in artifacts:
        return ArtifactState.CLEANED
    return ArtifactState.ABSENT


def _aggregate_postcondition(
    journals: tuple[LineageJournal, ...],
    occurrence: RequestedEffectOccurrence,
) -> PostconditionState:
    """Return only a request-wide postcondition justified by observed steps."""
    if occurrence is not RequestedEffectOccurrence.TRUE:
        return PostconditionState.UNKNOWN
    states = frozenset(journal.postcondition for journal in journals)
    if PostconditionState.UNKNOWN in states:
        return PostconditionState.UNKNOWN
    if PostconditionState.SUPERSEDED in states:
        return PostconditionState.SUPERSEDED
    return PostconditionState.ESTABLISHED


def _workspace_change(
    occurrence: RequestedEffectOccurrence, artifact: ArtifactState
) -> WorkspaceChange:
    """Derive current workspace change without conflating requested effects."""
    if occurrence is RequestedEffectOccurrence.TRUE:
        return WorkspaceChange.CHANGED
    if artifact in {ArtifactState.STAGED, ArtifactState.LEAKED}:
        return WorkspaceChange.CHANGED
    if (
        artifact is ArtifactState.UNKNOWN
        or occurrence is RequestedEffectOccurrence.UNKNOWN
    ):
        return WorkspaceChange.UNKNOWN
    return WorkspaceChange.UNCHANGED


def _validate_status_truth(
    status: PatchStatus, mutation: MutationState
) -> None:
    """Require the closed terminal status and mutation truth mapping."""
    expected = {
        PatchStatus.REJECTED: MutationState.NOT_COMMITTED,
        PatchStatus.DENIED: MutationState.NOT_COMMITTED,
        PatchStatus.APPROVAL_DENIED: MutationState.NOT_COMMITTED,
        PatchStatus.APPROVAL_UNAVAILABLE: MutationState.NOT_COMMITTED,
        PatchStatus.STALE: MutationState.NOT_COMMITTED,
        PatchStatus.CANCELLED: MutationState.NOT_COMMITTED,
        PatchStatus.COMMIT_FAILED: MutationState.NOT_COMMITTED,
        PatchStatus.COMMITTED: MutationState.COMMITTED,
        PatchStatus.PARTIAL: MutationState.PARTIALLY_COMMITTED,
        PatchStatus.INDETERMINATE: MutationState.INDETERMINATE,
    }
    if expected[status] is not mutation:
        raise PatchValidationError(
            "terminal status and mutation truth disagree"
        )


@dataclass(frozen=True, slots=True)
class DomainFacade:
    """Expose pure journal-to-outcome derivation without effect authority."""

    def pending(
        self,
        request_id: PatchRequestId,
        pending_operation_id: PatchPendingOperationId,
        correlation_id: PatchObserverCorrelationId,
    ) -> PatchPending:
        """Return one nonterminal public-safe settlement envelope."""
        return PatchPending(
            schema_version=1,
            pending_operation_id=pending_operation_id,
            request_id=request_id,
            correlation_id=correlation_id,
            lifecycle=LifecyclePhase.SETTLEMENT_PENDING,
        )

    def settle(
        self,
        plan: MutationPlan,
        journals: tuple[LineageJournal, ...],
        diagnostic: PatchDiagnostic,
    ) -> PatchResult:
        """Derive one terminal result and its sole completion event truth."""
        truth = derive_commit_truth(plan, journals)
        status = _status_for_truth(truth.mutation_state)
        return PatchResult(
            schema_version=1,
            request_id=plan.request.request_id,
            plan_id=plan.plan_id,
            lifecycle=LifecyclePhase.REQUEST_COMPLETED,
            status=status,
            truth=truth,
            diagnostic=diagnostic,
        )

    def settle_with_event(
        self,
        plan: MutationPlan,
        journals: tuple[LineageJournal, ...],
        diagnostic: PatchDiagnostic,
        event_id: PatchEventId,
        observer_id: PatchObserverId,
        correlation_id: PatchObserverCorrelationId,
        sequence: SequenceNumber,
    ) -> tuple[PatchResult, PatchLifecycleEvent]:
        """Derive one terminal result and its sole terminal event record."""
        result = self.settle(plan, journals, diagnostic)
        event = PatchLifecycleEvent(
            schema_version=1,
            event_id=event_id,
            observer_id=observer_id,
            correlation_id=correlation_id,
            request_id=plan.request.request_id,
            sequence=sequence,
            lifecycle=LifecyclePhase.REQUEST_COMPLETED,
        )
        return result, event


def _status_for_truth(mutation: MutationState) -> PatchStatus:
    """Return the commit-stage terminal status for exact journal truth."""
    match mutation:
        case MutationState.NOT_COMMITTED:
            return PatchStatus.COMMIT_FAILED
        case MutationState.COMMITTED:
            return PatchStatus.COMMITTED
        case MutationState.PARTIALLY_COMMITTED:
            return PatchStatus.PARTIAL
        case MutationState.INDETERMINATE:
            return PatchStatus.INDETERMINATE
