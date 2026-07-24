"""Define portable durable-continuation values and runtime resolution."""

from ..types import JsonObject, JsonValue, MutableJsonValue
from .codec import (
    decode_continuation_snapshot,
    decode_execution_origin,
    encode_continuation_snapshot,
    encode_execution_origin,
)
from .entities import (
    CapabilityRevision,
    ContinuationId,
    ContinuationRevisionBinding,
    ContinuationSnapshot,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputRequestId,
    ModelCallId,
    ModelConfigRevision,
    ModelId,
    ProviderConfigRevision,
    ProviderFamilyName,
    ProviderIdempotencyKey,
    StateRevision,
    _freeze_snapshot_object,
)
from .error import (
    InputErrorCode,
    InputSnapshotError,
    InputValidationError,
)
from .validation import (
    MAX_STATE_REVISION,
    validate_aware_datetime,
    validate_int,
    validate_opaque_id,
    validate_state_revision,
)

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from inspect import isawaitable
from json import JSONDecodeError, dumps, loads
from re import compile as compile_pattern
from typing import NewType, Protocol, TypeAlias, cast, final

PORTABLE_CONTINUATION_VERSION = 1
PORTABLE_CONTINUATION_MAX_UTF8_BYTES = 4_194_304

ContinuationClaimOwnerId = NewType("ContinuationClaimOwnerId", str)
ContinuationDispatchId = NewType("ContinuationDispatchId", str)
ContinuationFencingToken = NewType("ContinuationFencingToken", int)
ContinuationStoreRevision = NewType("ContinuationStoreRevision", int)

ProviderContinuationSnapshot: TypeAlias = ContinuationSnapshot

_SHA256_PATTERN = compile_pattern(r"^[0-9a-f]{64}$")
_PORTABLE_FIELDS = {
    "version",
    "continuation_id",
    "request_id",
    "origin",
    "provider_call_id",
    "provider_call_correlation_id",
    "definition",
    "operation_cursor",
    "generation_settings",
    "transcript",
    "observations",
    "provider_snapshot",
    "revision_binding",
    "interaction_count",
    "tool_loop_count",
    "stream_sequence",
    "state_revision",
    "store_revision",
    "created_at",
    "updated_at",
    "expires_at",
    "claim",
    "fencing_token",
    "dispatch",
    "completion",
    "content_sha256",
}


def derive_continuation_dispatch_id(
    continuation_id: ContinuationId,
) -> ContinuationDispatchId:
    """Derive one stable logical dispatch from a continuation identity."""
    continuation = validate_opaque_id(
        continuation_id,
        "continuation.continuation_id",
    )
    encoded = dumps(
        {"continuation_id": continuation},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return ContinuationDispatchId(
        f"task-resume-{sha256(encoded.encode('utf-8')).hexdigest()}"
    )


def derive_provider_idempotency_key(
    continuation_id: ContinuationId,
    dispatch_id: ContinuationDispatchId,
) -> ProviderIdempotencyKey:
    """Derive one retry-stable provider key from exact durable identity."""
    continuation = validate_opaque_id(
        continuation_id,
        "continuation.continuation_id",
    )
    dispatch = validate_opaque_id(
        dispatch_id,
        "continuation.dispatch.dispatch_id",
    )
    encoded = dumps(
        {
            "continuation_id": continuation,
            "dispatch_id": dispatch,
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return ProviderIdempotencyKey(
        f"task-input-{sha256(encoded.encode('utf-8')).hexdigest()}"
    )


class ContinuationClaimState(StrEnum):
    """Identify one durable provider-dispatch claim state."""

    UNCLAIMED = "unclaimed"
    CLAIMED_PRE_DISPATCH = "claimed_pre_dispatch"
    DISPATCHED_AMBIGUOUS = "dispatched_ambiguous"
    COMPLETED = "completed"
    FAILED_SAFE_TO_RETRY = "failed_safe_to_retry"


class DurableContinuationResumeState(StrEnum):
    """Identify process-local progress around one durable dispatch."""

    ADMITTED = "admitted"
    DISPATCHING = "dispatching"
    DISPATCHED = "dispatched"
    AMBIGUOUS = "ambiguous"
    COMPLETED = "completed"
    RELEASED = "released"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationClaim:
    """Carry durable claim ownership without embedding a live worker."""

    state: ContinuationClaimState = ContinuationClaimState.UNCLAIMED
    owner_id: ContinuationClaimOwnerId | None = None
    lease_expires_at: datetime | None = None
    attempt: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.state, ContinuationClaimState):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "continuation.claim.state",
                "claim state must be a continuation claim state",
            )
        if self.owner_id is not None:
            object.__setattr__(
                self,
                "owner_id",
                ContinuationClaimOwnerId(
                    validate_opaque_id(
                        self.owner_id,
                        "continuation.claim.owner_id",
                    )
                ),
            )
        if self.lease_expires_at is not None:
            object.__setattr__(
                self,
                "lease_expires_at",
                validate_aware_datetime(
                    self.lease_expires_at,
                    "continuation.claim.lease_expires_at",
                ),
            )
        object.__setattr__(
            self,
            "attempt",
            validate_int(
                self.attempt,
                "continuation.claim.attempt",
                minimum=0,
                maximum=MAX_STATE_REVISION,
            ),
        )
        owns_dispatch = self.state in {
            ContinuationClaimState.CLAIMED_PRE_DISPATCH,
            ContinuationClaimState.DISPATCHED_AMBIGUOUS,
        }
        if owns_dispatch != (self.owner_id is not None):
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "continuation.claim.owner_id",
                "claim owner does not match the claim state",
            )
        has_lease = self.state is ContinuationClaimState.CLAIMED_PRE_DISPATCH
        if has_lease != (self.lease_expires_at is not None):
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "continuation.claim.lease_expires_at",
                "claim lease does not match the claim state",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationDispatch:
    """Identify one logical provider dispatch and its stable retry key."""

    dispatch_id: ContinuationDispatchId
    provider_idempotency_key: ProviderIdempotencyKey
    marked_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dispatch_id",
            ContinuationDispatchId(
                validate_opaque_id(
                    self.dispatch_id,
                    "continuation.dispatch.dispatch_id",
                )
            ),
        )
        object.__setattr__(
            self,
            "provider_idempotency_key",
            ProviderIdempotencyKey(
                validate_opaque_id(
                    self.provider_idempotency_key,
                    "continuation.dispatch.provider_idempotency_key",
                    maximum_characters=256,
                    maximum_bytes=1_024,
                )
            ),
        )
        object.__setattr__(
            self,
            "marked_at",
            validate_aware_datetime(
                self.marked_at,
                "continuation.dispatch.marked_at",
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationCompletion:
    """Record content-safe completion metadata after durable dispatch."""

    completed_at: datetime
    result_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "completed_at",
            validate_aware_datetime(
                self.completed_at,
                "continuation.completion.completed_at",
            ),
        )
        object.__setattr__(
            self,
            "result_digest",
            _validate_sha256(
                self.result_digest,
                "continuation.completion.result_digest",
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationCompletionCommand:
    """Fence one completion or atomic successor suspension."""

    continuation_id: ContinuationId
    expected_store_revision: ContinuationStoreRevision
    owner_id: ContinuationClaimOwnerId
    fencing_token: ContinuationFencingToken
    result_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "continuation_id",
            ContinuationId(
                validate_opaque_id(
                    self.continuation_id,
                    "continuation_completion.continuation_id",
                )
            ),
        )
        object.__setattr__(
            self,
            "expected_store_revision",
            ContinuationStoreRevision(
                validate_state_revision(
                    self.expected_store_revision,
                    "continuation_completion.expected_store_revision",
                )
            ),
        )
        object.__setattr__(
            self,
            "owner_id",
            ContinuationClaimOwnerId(
                validate_opaque_id(
                    self.owner_id,
                    "continuation_completion.owner_id",
                )
            ),
        )
        object.__setattr__(
            self,
            "fencing_token",
            ContinuationFencingToken(
                validate_state_revision(
                    self.fencing_token,
                    "continuation_completion.fencing_token",
                )
            ),
        )
        object.__setattr__(
            self,
            "result_digest",
            _validate_sha256(
                self.result_digest,
                "continuation_completion.result_digest",
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationRejectionCommand:
    """Fence one deterministic pre-dispatch continuation rejection."""

    continuation_id: ContinuationId
    expected_store_revision: ContinuationStoreRevision
    owner_id: ContinuationClaimOwnerId
    fencing_token: ContinuationFencingToken
    result_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "continuation_id",
            ContinuationId(
                validate_opaque_id(
                    self.continuation_id,
                    "continuation_rejection.continuation_id",
                )
            ),
        )
        object.__setattr__(
            self,
            "expected_store_revision",
            ContinuationStoreRevision(
                validate_state_revision(
                    self.expected_store_revision,
                    "continuation_rejection.expected_store_revision",
                )
            ),
        )
        object.__setattr__(
            self,
            "owner_id",
            ContinuationClaimOwnerId(
                validate_opaque_id(
                    self.owner_id,
                    "continuation_rejection.owner_id",
                )
            ),
        )
        object.__setattr__(
            self,
            "fencing_token",
            ContinuationFencingToken(
                validate_state_revision(
                    self.fencing_token,
                    "continuation_rejection.fencing_token",
                )
            ),
        )
        object.__setattr__(
            self,
            "result_digest",
            _validate_sha256(
                self.result_digest,
                "continuation_rejection.result_digest",
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PortableContinuation:
    """Store only portable state needed to reconstruct one continuation."""

    continuation_id: ContinuationId
    request_id: InputRequestId
    origin: ExecutionOrigin
    provider_call_id: ModelCallId
    provider_call_correlation_id: str
    definition: ExecutionDefinitionRef
    operation_cursor: int
    generation_settings: Mapping[str, JsonValue]
    transcript: tuple[Mapping[str, JsonValue], ...]
    observations: tuple[Mapping[str, JsonValue], ...]
    revision_binding: ContinuationRevisionBinding
    interaction_count: int
    tool_loop_count: int
    stream_sequence: int
    state_revision: StateRevision
    store_revision: ContinuationStoreRevision
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    claim: ContinuationClaim = field(default_factory=ContinuationClaim)
    fencing_token: ContinuationFencingToken = ContinuationFencingToken(0)
    provider_snapshot: ProviderContinuationSnapshot | None = None
    dispatch: ContinuationDispatch | None = None
    completion: ContinuationCompletion | None = None
    version: int = PORTABLE_CONTINUATION_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.version) is not int
            or self.version != PORTABLE_CONTINUATION_VERSION
        ):
            raise InputValidationError(
                InputErrorCode.SNAPSHOT_UNSUPPORTED,
                "continuation.version",
                "portable continuation version is unsupported",
            )
        object.__setattr__(
            self,
            "continuation_id",
            ContinuationId(
                validate_opaque_id(
                    self.continuation_id,
                    "continuation.continuation_id",
                )
            ),
        )
        object.__setattr__(
            self,
            "request_id",
            InputRequestId(
                validate_opaque_id(
                    self.request_id,
                    "continuation.request_id",
                )
            ),
        )
        if type(self.origin) is not ExecutionOrigin:
            _invalid_type("continuation.origin", "an execution origin")
        object.__setattr__(
            self,
            "provider_call_id",
            ModelCallId(
                validate_opaque_id(
                    self.provider_call_id,
                    "continuation.provider_call_id",
                )
            ),
        )
        object.__setattr__(
            self,
            "provider_call_correlation_id",
            validate_opaque_id(
                self.provider_call_correlation_id,
                "continuation.provider_call_correlation_id",
                maximum_characters=256,
                maximum_bytes=1_024,
            ),
        )
        if type(self.definition) is not ExecutionDefinitionRef:
            _invalid_type(
                "continuation.definition",
                "an execution definition reference",
            )
        if self.origin.definition != self.definition:
            _correlation_error(
                "continuation.definition",
                "definition does not match the execution origin",
            )
        if self.origin.model_call_id != self.provider_call_id:
            _correlation_error(
                "continuation.provider_call_id",
                "provider call does not match the execution origin",
            )
        object.__setattr__(
            self,
            "operation_cursor",
            validate_int(
                self.operation_cursor,
                "continuation.operation_cursor",
                minimum=0,
                maximum=MAX_STATE_REVISION,
            ),
        )
        object.__setattr__(
            self,
            "generation_settings",
            _freeze_json_object(
                self.generation_settings,
                "continuation.generation_settings",
            ),
        )
        for name in ("transcript", "observations"):
            object.__setattr__(
                self,
                name,
                _freeze_json_records(
                    getattr(self, name),
                    f"continuation.{name}",
                ),
            )
        if type(self.revision_binding) is not ContinuationRevisionBinding:
            _invalid_type(
                "continuation.revision_binding",
                "a continuation revision binding",
            )
        for name in (
            "interaction_count",
            "tool_loop_count",
            "stream_sequence",
        ):
            object.__setattr__(
                self,
                name,
                validate_int(
                    getattr(self, name),
                    f"continuation.{name}",
                    minimum=0,
                    maximum=MAX_STATE_REVISION,
                ),
            )
        object.__setattr__(
            self,
            "state_revision",
            StateRevision(
                validate_state_revision(
                    self.state_revision,
                    "continuation.state_revision",
                )
            ),
        )
        object.__setattr__(
            self,
            "store_revision",
            ContinuationStoreRevision(
                validate_state_revision(
                    self.store_revision,
                    "continuation.store_revision",
                )
            ),
        )
        for name in ("created_at", "updated_at", "expires_at"):
            object.__setattr__(
                self,
                name,
                validate_aware_datetime(
                    getattr(self, name),
                    f"continuation.{name}",
                ),
            )
        if (
            self.updated_at < self.created_at
            or self.expires_at <= self.created_at
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "continuation.timestamps",
                "creation must precede updates and expiry",
            )
        if type(self.claim) is not ContinuationClaim:
            _invalid_type("continuation.claim", "a continuation claim")
        object.__setattr__(
            self,
            "fencing_token",
            ContinuationFencingToken(
                validate_state_revision(
                    self.fencing_token,
                    "continuation.fencing_token",
                )
            ),
        )
        if (
            self.dispatch is not None
            and type(self.dispatch) is not ContinuationDispatch
        ):
            _invalid_type("continuation.dispatch", "dispatch metadata")
        if (
            self.completion is not None
            and type(self.completion) is not ContinuationCompletion
        ):
            _invalid_type("continuation.completion", "completion metadata")
        if self.provider_snapshot is not None:
            self._validate_provider_snapshot()
        self._validate_claim_metadata()

    def _validate_provider_snapshot(self) -> None:
        snapshot = self.provider_snapshot
        assert snapshot is not None
        if type(snapshot) is not ContinuationSnapshot:
            _invalid_type(
                "continuation.provider_snapshot",
                "a provider continuation snapshot",
            )
        if snapshot.revision_binding != self.revision_binding:
            raise InputValidationError(
                InputErrorCode.SNAPSHOT_REVISION_DRIFT,
                "continuation.provider_snapshot.revision_binding",
                "provider snapshot revision does not match the continuation",
            )
        if snapshot.model_call_id != self.provider_call_id:
            _correlation_error(
                "continuation.provider_snapshot.model_call_id",
                "provider snapshot call does not match the continuation",
            )
        if (
            self.dispatch is not None
            and snapshot.provider_idempotency_key
            != self.dispatch.provider_idempotency_key
        ):
            _correlation_error(
                "continuation.provider_snapshot.provider_idempotency_key",
                "provider snapshot retry key does not match the dispatch",
            )

    def _validate_claim_metadata(self) -> None:
        state = self.claim.state
        dispatch_required = state in {
            ContinuationClaimState.CLAIMED_PRE_DISPATCH,
            ContinuationClaimState.DISPATCHED_AMBIGUOUS,
            ContinuationClaimState.COMPLETED,
            ContinuationClaimState.FAILED_SAFE_TO_RETRY,
        }
        if dispatch_required != (self.dispatch is not None):
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "continuation.dispatch",
                "dispatch metadata does not match the claim state",
            )
        completed = state is ContinuationClaimState.COMPLETED
        if completed != (self.completion is not None):
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "continuation.completion",
                "completion metadata does not match the claim state",
            )
        if (
            self.dispatch is not None
            and self.dispatch.marked_at < self.created_at
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "continuation.dispatch.marked_at",
                "dispatch metadata predates the continuation",
            )
        if self.completion is not None:
            assert self.dispatch is not None
            if (
                self.completion.completed_at < self.dispatch.marked_at
                or self.completion.completed_at > self.updated_at
            ):
                raise InputValidationError(
                    InputErrorCode.INVALID_FORMAT,
                    "continuation.completion.completed_at",
                    "completion timestamp is outside the continuation",
                )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationClaimReceipt:
    """Return one claimed portable continuation and its fence."""

    continuation: PortableContinuation
    fencing_token: ContinuationFencingToken

    def __post_init__(self) -> None:
        if type(self.continuation) is not PortableContinuation:
            _invalid_type(
                "continuation_claim.continuation",
                "a portable continuation",
            )
        if self.continuation.fencing_token != self.fencing_token:
            _correlation_error(
                "continuation_claim.fencing_token",
                "claim receipt does not match the continuation fence",
            )
        if (
            self.continuation.claim.state
            is not ContinuationClaimState.CLAIMED_PRE_DISPATCH
        ):
            raise InputValidationError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "continuation_claim.continuation.claim.state",
                "claim receipt must contain a pre-dispatch claim",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DurableContinuationRecord:
    """Bind one portable continuation to its durable task checkpoint."""

    continuation: PortableContinuation
    task_run_id: str | None = None
    checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.continuation) is not PortableContinuation:
            _invalid_type(
                "continuation_record.continuation",
                "a portable continuation",
            )
        if self.task_run_id is not None:
            task_run_id = validate_opaque_id(
                self.task_run_id,
                "continuation_record.task_run_id",
            )
            object.__setattr__(self, "task_run_id", task_run_id)
            if task_run_id != str(self.continuation.origin.run_id):
                _correlation_error(
                    "continuation_record.task_run_id",
                    "task run does not match the continuation origin",
                )
        if self.checkpoint_id is not None:
            checkpoint_id = validate_opaque_id(
                self.checkpoint_id,
                "continuation_record.checkpoint_id",
            )
            object.__setattr__(self, "checkpoint_id", checkpoint_id)
            if self.task_run_id is None:
                _correlation_error(
                    "continuation_record.checkpoint_id",
                    "checkpoint requires a bound task run",
                )


class ProviderContinuationSnapshotAdapter(Protocol):
    """Export and import validated provider-owned replay state."""

    def export_continuation_snapshot(
        self,
        *,
        revision_binding: ContinuationRevisionBinding,
        model_call_id: ModelCallId,
        provider_idempotency_key: ProviderIdempotencyKey,
        provider_call_correlation_id: str,
    ) -> ProviderContinuationSnapshot:
        """Export one JSON-safe provider replay capsule."""
        ...

    def import_continuation_snapshot(
        self,
        snapshot: ProviderContinuationSnapshot,
        *,
        expected_binding: ContinuationRevisionBinding,
        provider_call_correlation_id: str,
    ) -> None:
        """Import one validated capsule into fresh adapter state."""
        ...

    def validate_continuation_snapshot_call(
        self,
        snapshot: ProviderContinuationSnapshot,
        *,
        expected_binding: ContinuationRevisionBinding,
        provider_call_correlation_id: str,
        expected_provider_name: str,
        expected_arguments: Mapping[str, JsonValue],
    ) -> None:
        """Validate the exact reserved call captured by one capsule."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedContinuationRuntime:
    """Carry fresh live components plus their validated portable identity."""

    definition: ExecutionDefinitionRef
    revision_binding: ContinuationRevisionBinding
    runtime: object = field(repr=False)
    operation: object = field(repr=False)
    model: object = field(repr=False)
    tools: object = field(repr=False)
    capabilities: object = field(repr=False)
    credentials_reloaded_from_trusted_config: bool

    def __post_init__(self) -> None:
        if type(self.definition) is not ExecutionDefinitionRef:
            _invalid_type(
                "continuation_runtime.definition",
                "an execution definition reference",
            )
        if type(self.revision_binding) is not ContinuationRevisionBinding:
            _invalid_type(
                "continuation_runtime.revision_binding",
                "a continuation revision binding",
            )
        for name in ("runtime", "operation", "model", "tools", "capabilities"):
            if getattr(self, name) is None:
                _invalid_type(
                    f"continuation_runtime.{name}",
                    "a fresh live runtime component",
                )
        if self.credentials_reloaded_from_trusted_config is not True:
            raise InputValidationError(
                InputErrorCode.UNAVAILABLE,
                "continuation_runtime.credentials",
                "credentials were not loaded from trusted configuration",
            )


class TrustedContinuationRuntimeLoader(Protocol):
    """Load a fresh continuation runtime from trusted configuration."""

    trusted_continuation_runtime_loader: bool

    async def load_continuation_runtime(
        self,
        definition: ExecutionDefinitionRef,
        revision_binding: ContinuationRevisionBinding,
    ) -> ResolvedContinuationRuntime:
        """Load exact agent, operation, model, tools, and capabilities."""
        ...


@final
class ContinuationRuntimeResolver:
    """Resolve portable definitions without retaining an orchestrator."""

    def __init__(
        self,
        loader: TrustedContinuationRuntimeLoader,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if getattr(
            loader, "trusted_continuation_runtime_loader", False
        ) is not True or not callable(
            getattr(loader, "load_continuation_runtime", None)
        ):
            raise InputValidationError(
                InputErrorCode.UNAVAILABLE,
                "continuation_runtime.loader",
                "a trusted async continuation runtime loader is required",
            )
        if clock is not None and not callable(clock):
            _invalid_type("continuation_runtime.clock", "a clock callable")
        self._loader = loader
        self._clock = clock or (lambda: datetime.now(UTC))

    async def resolve(
        self,
        continuation: PortableContinuation,
    ) -> ResolvedContinuationRuntime:
        """Load and validate a fresh exact runtime for one continuation."""
        if type(continuation) is not PortableContinuation:
            _invalid_type(
                "continuation_runtime.continuation",
                "a portable continuation",
            )
        now = validate_aware_datetime(
            self._clock(),
            "continuation_runtime.clock",
        )
        if now >= continuation.expires_at:
            raise InputValidationError(
                InputErrorCode.EXPIRED,
                "continuation.expires_at",
                "continuation has expired",
            )
        pending = self._loader.load_continuation_runtime(
            continuation.definition,
            continuation.revision_binding,
        )
        if not isawaitable(pending):
            raise InputValidationError(
                InputErrorCode.UNAVAILABLE,
                "continuation_runtime.loader",
                "continuation runtime loader must be asynchronous",
            )
        resolved = await pending
        if type(resolved) is not ResolvedContinuationRuntime:
            _invalid_type(
                "continuation_runtime.result",
                "a resolved continuation runtime",
            )
        if resolved.definition != continuation.definition:
            raise InputValidationError(
                InputErrorCode.SNAPSHOT_REVISION_DRIFT,
                "continuation_runtime.definition",
                "resolved execution definition has drifted",
            )
        _validate_revision_binding(
            resolved.revision_binding,
            continuation.revision_binding,
            path="continuation_runtime.revision_binding",
        )
        return resolved


def encode_portable_continuation(
    continuation: PortableContinuation,
) -> str:
    """Encode one portable continuation as canonical strict JSON."""
    if type(continuation) is not PortableContinuation:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "continuation",
            "value must be a portable continuation",
        )
    payload = _portable_payload(continuation)
    payload["content_sha256"] = _portable_digest_payload(payload)
    encoded = _canonical_json(payload)
    if len(encoded.encode("utf-8")) > PORTABLE_CONTINUATION_MAX_UTF8_BYTES:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "continuation",
            "portable continuation exceeds its byte bound",
        )
    return encoded


def decode_portable_continuation(
    value: str | bytes,
    *,
    expected_binding: ContinuationRevisionBinding,
) -> PortableContinuation:
    """Decode a portable continuation and reject identity or revision drift."""
    text = _continuation_text(value)
    try:
        raw = loads(text, object_pairs_hook=_unique_object)
    except (JSONDecodeError, UnicodeError, ValueError) as exc:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "continuation",
            "portable continuation is invalid JSON",
        ) from exc
    if type(raw) is not dict:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "continuation",
            "portable continuation must be a JSON object",
        )
    item = cast(dict[str, object], raw)
    if set(item) != _PORTABLE_FIELDS:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "continuation",
            "portable continuation fields do not match its schema",
        )
    if _canonical_json(cast(JsonObject, item)) != text:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "continuation",
            "portable continuation must use canonical JSON",
        )
    version = _integer(item["version"], "continuation.version")
    if version != PORTABLE_CONTINUATION_VERSION:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_UNSUPPORTED,
            "continuation.version",
            "portable continuation version is unsupported",
        )
    digest = _string(
        item["content_sha256"],
        "continuation.content_sha256",
    )
    try:
        _validate_sha256(digest, "continuation.content_sha256")
    except InputValidationError as exc:
        raise InputSnapshotError(exc.code, exc.path, exc.safe_message) from exc
    digest_payload = dict(item)
    del digest_payload["content_sha256"]
    if digest != _portable_digest_payload(cast(JsonObject, digest_payload)):
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "continuation.content_sha256",
            "portable continuation content hash does not match",
        )
    binding = _decode_revision_binding(item["revision_binding"])
    _validate_revision_binding(
        binding,
        expected_binding,
        path="continuation.revision_binding",
    )
    provider_snapshot_value = item["provider_snapshot"]
    provider_snapshot = (
        None
        if provider_snapshot_value is None
        else decode_continuation_snapshot(
            _canonical_json(
                cast(JsonObject, _object(provider_snapshot_value, "snapshot"))
            ),
            expected_binding=expected_binding,
        )
    )
    try:
        return PortableContinuation(
            version=version,
            continuation_id=ContinuationId(
                _string(
                    item["continuation_id"],
                    "continuation.continuation_id",
                )
            ),
            request_id=InputRequestId(
                _string(item["request_id"], "continuation.request_id")
            ),
            origin=decode_execution_origin(item["origin"]),
            provider_call_id=ModelCallId(
                _string(
                    item["provider_call_id"],
                    "continuation.provider_call_id",
                )
            ),
            provider_call_correlation_id=_string(
                item["provider_call_correlation_id"],
                "continuation.provider_call_correlation_id",
            ),
            definition=_decode_definition(item["definition"]),
            operation_cursor=_integer(
                item["operation_cursor"],
                "continuation.operation_cursor",
            ),
            generation_settings=_json_mapping(
                item["generation_settings"],
                "continuation.generation_settings",
            ),
            transcript=_json_records(
                item["transcript"],
                "continuation.transcript",
            ),
            observations=_json_records(
                item["observations"],
                "continuation.observations",
            ),
            provider_snapshot=provider_snapshot,
            revision_binding=binding,
            interaction_count=_integer(
                item["interaction_count"],
                "continuation.interaction_count",
            ),
            tool_loop_count=_integer(
                item["tool_loop_count"],
                "continuation.tool_loop_count",
            ),
            stream_sequence=_integer(
                item["stream_sequence"],
                "continuation.stream_sequence",
            ),
            state_revision=StateRevision(
                _integer(
                    item["state_revision"],
                    "continuation.state_revision",
                )
            ),
            store_revision=ContinuationStoreRevision(
                _integer(
                    item["store_revision"],
                    "continuation.store_revision",
                )
            ),
            created_at=_datetime(
                item["created_at"],
                "continuation.created_at",
            ),
            updated_at=_datetime(
                item["updated_at"],
                "continuation.updated_at",
            ),
            expires_at=_datetime(
                item["expires_at"],
                "continuation.expires_at",
            ),
            claim=_decode_claim(item["claim"]),
            fencing_token=ContinuationFencingToken(
                _integer(
                    item["fencing_token"],
                    "continuation.fencing_token",
                )
            ),
            dispatch=_decode_dispatch(item["dispatch"]),
            completion=_decode_completion(item["completion"]),
        )
    except InputSnapshotError:
        raise
    except InputValidationError as exc:
        raise InputSnapshotError(exc.code, exc.path, exc.safe_message) from exc


def portable_continuation_digest(
    continuation: PortableContinuation,
) -> str:
    """Return the canonical semantic digest of a portable continuation."""
    if type(continuation) is not PortableContinuation:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "continuation",
            "value must be a portable continuation",
        )
    return _portable_digest_payload(_portable_payload(continuation))


def _portable_payload(continuation: PortableContinuation) -> JsonObject:
    return {
        "version": continuation.version,
        "continuation_id": str(continuation.continuation_id),
        "request_id": str(continuation.request_id),
        "origin": encode_execution_origin(continuation.origin),
        "provider_call_id": str(continuation.provider_call_id),
        "provider_call_correlation_id": (
            continuation.provider_call_correlation_id
        ),
        "definition": _encode_definition(continuation.definition),
        "operation_cursor": continuation.operation_cursor,
        "generation_settings": _mutable_json_mapping(
            continuation.generation_settings
        ),
        "transcript": [
            _mutable_json_mapping(item) for item in continuation.transcript
        ],
        "observations": [
            _mutable_json_mapping(item) for item in continuation.observations
        ],
        "provider_snapshot": (
            None
            if continuation.provider_snapshot is None
            else cast(
                JsonObject,
                loads(
                    encode_continuation_snapshot(
                        continuation.provider_snapshot
                    )
                ),
            )
        ),
        "revision_binding": _encode_revision_binding(
            continuation.revision_binding
        ),
        "interaction_count": continuation.interaction_count,
        "tool_loop_count": continuation.tool_loop_count,
        "stream_sequence": continuation.stream_sequence,
        "state_revision": continuation.state_revision,
        "store_revision": continuation.store_revision,
        "created_at": continuation.created_at.isoformat(),
        "updated_at": continuation.updated_at.isoformat(),
        "expires_at": continuation.expires_at.isoformat(),
        "claim": _encode_claim(continuation.claim),
        "fencing_token": int(continuation.fencing_token),
        "dispatch": (
            None
            if continuation.dispatch is None
            else _encode_dispatch(continuation.dispatch)
        ),
        "completion": (
            None
            if continuation.completion is None
            else _encode_completion(continuation.completion)
        ),
    }


def _portable_digest_payload(payload: JsonObject) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _encode_definition(value: ExecutionDefinitionRef) -> JsonObject:
    return {
        "agent_definition_locator": value.agent_definition_locator,
        "agent_definition_revision": value.agent_definition_revision,
        "operation_id": value.operation_id,
        "operation_index": value.operation_index,
        "model_config_reference": value.model_config_reference,
        "tool_revision": value.tool_revision,
        "capability_revision": value.capability_revision,
    }


def _decode_definition(value: object) -> ExecutionDefinitionRef:
    item = _object(value, "continuation.definition")
    expected = {
        "agent_definition_locator",
        "agent_definition_revision",
        "operation_id",
        "operation_index",
        "model_config_reference",
        "tool_revision",
        "capability_revision",
    }
    _require_fields(item, expected, "continuation.definition")
    return ExecutionDefinitionRef(
        agent_definition_locator=_string(
            item["agent_definition_locator"],
            "continuation.definition.agent_definition_locator",
        ),
        agent_definition_revision=_string(
            item["agent_definition_revision"],
            "continuation.definition.agent_definition_revision",
        ),
        operation_id=_string(
            item["operation_id"],
            "continuation.definition.operation_id",
        ),
        operation_index=_integer(
            item["operation_index"],
            "continuation.definition.operation_index",
        ),
        model_config_reference=_string(
            item["model_config_reference"],
            "continuation.definition.model_config_reference",
        ),
        tool_revision=_string(
            item["tool_revision"],
            "continuation.definition.tool_revision",
        ),
        capability_revision=_string(
            item["capability_revision"],
            "continuation.definition.capability_revision",
        ),
    )


def _encode_revision_binding(
    value: ContinuationRevisionBinding,
) -> JsonObject:
    return {
        "provider_family": str(value.provider_family),
        "model_id": str(value.model_id),
        "provider_config_revision": str(value.provider_config_revision),
        "model_config_revision": str(value.model_config_revision),
        "capability_revision": str(value.capability_revision),
    }


def _decode_revision_binding(value: object) -> ContinuationRevisionBinding:
    item = _object(value, "continuation.revision_binding")
    expected = {
        "provider_family",
        "model_id",
        "provider_config_revision",
        "model_config_revision",
        "capability_revision",
    }
    _require_fields(item, expected, "continuation.revision_binding")
    try:
        return ContinuationRevisionBinding(
            provider_family=ProviderFamilyName(
                _string(
                    item["provider_family"],
                    "continuation.revision_binding.provider_family",
                )
            ),
            model_id=ModelId(
                _string(
                    item["model_id"],
                    "continuation.revision_binding.model_id",
                )
            ),
            provider_config_revision=ProviderConfigRevision(
                _string(
                    item["provider_config_revision"],
                    "continuation.revision_binding.provider_config_revision",
                )
            ),
            model_config_revision=ModelConfigRevision(
                _string(
                    item["model_config_revision"],
                    "continuation.revision_binding.model_config_revision",
                )
            ),
            capability_revision=CapabilityRevision(
                _string(
                    item["capability_revision"],
                    "continuation.revision_binding.capability_revision",
                )
            ),
        )
    except InputValidationError as exc:
        raise InputSnapshotError(exc.code, exc.path, exc.safe_message) from exc


def _validate_revision_binding(
    actual: ContinuationRevisionBinding,
    expected: ContinuationRevisionBinding,
    *,
    path: str,
) -> None:
    if type(expected) is not ContinuationRevisionBinding:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            path,
            "expected binding must be a typed revision binding",
        )
    if (
        actual.provider_family != expected.provider_family
        or actual.model_id != expected.model_id
    ):
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_PROVIDER_UNAVAILABLE,
            path,
            "continuation provider or model is unavailable",
        )
    if actual != expected:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_REVISION_DRIFT,
            path,
            "continuation configuration revision has drifted",
        )


def _encode_claim(value: ContinuationClaim) -> JsonObject:
    return {
        "state": value.state.value,
        "owner_id": None if value.owner_id is None else str(value.owner_id),
        "lease_expires_at": (
            None
            if value.lease_expires_at is None
            else value.lease_expires_at.isoformat()
        ),
        "attempt": value.attempt,
    }


def _decode_claim(value: object) -> ContinuationClaim:
    item = _object(value, "continuation.claim")
    _require_fields(
        item,
        {"state", "owner_id", "lease_expires_at", "attempt"},
        "continuation.claim",
    )
    try:
        state = ContinuationClaimState(
            _string(item["state"], "continuation.claim.state")
        )
    except ValueError as exc:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "continuation.claim.state",
            "continuation claim state is unsupported",
        ) from exc
    owner_value = item["owner_id"]
    lease_value = item["lease_expires_at"]
    return ContinuationClaim(
        state=state,
        owner_id=(
            None
            if owner_value is None
            else ContinuationClaimOwnerId(
                _string(owner_value, "continuation.claim.owner_id")
            )
        ),
        lease_expires_at=(
            None
            if lease_value is None
            else _datetime(
                lease_value,
                "continuation.claim.lease_expires_at",
            )
        ),
        attempt=_integer(item["attempt"], "continuation.claim.attempt"),
    )


def _encode_dispatch(value: ContinuationDispatch) -> JsonObject:
    return {
        "dispatch_id": str(value.dispatch_id),
        "provider_idempotency_key": str(value.provider_idempotency_key),
        "marked_at": value.marked_at.isoformat(),
    }


def _decode_dispatch(value: object) -> ContinuationDispatch | None:
    if value is None:
        return None
    item = _object(value, "continuation.dispatch")
    _require_fields(
        item,
        {"dispatch_id", "provider_idempotency_key", "marked_at"},
        "continuation.dispatch",
    )
    return ContinuationDispatch(
        dispatch_id=ContinuationDispatchId(
            _string(
                item["dispatch_id"],
                "continuation.dispatch.dispatch_id",
            )
        ),
        provider_idempotency_key=ProviderIdempotencyKey(
            _string(
                item["provider_idempotency_key"],
                "continuation.dispatch.provider_idempotency_key",
            )
        ),
        marked_at=_datetime(
            item["marked_at"],
            "continuation.dispatch.marked_at",
        ),
    )


def _encode_completion(value: ContinuationCompletion) -> JsonObject:
    return {
        "completed_at": value.completed_at.isoformat(),
        "result_digest": value.result_digest,
    }


def _decode_completion(value: object) -> ContinuationCompletion | None:
    if value is None:
        return None
    item = _object(value, "continuation.completion")
    _require_fields(
        item,
        {"completed_at", "result_digest"},
        "continuation.completion",
    )
    return ContinuationCompletion(
        completed_at=_datetime(
            item["completed_at"],
            "continuation.completion.completed_at",
        ),
        result_digest=_string(
            item["result_digest"],
            "continuation.completion.result_digest",
        ),
    )


def _freeze_json_object(
    value: object,
    path: str,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        _invalid_type(path, "a JSON object")
    return _freeze_snapshot_object(cast(Mapping[str, object], value), path)


def _freeze_json_records(
    value: object,
    path: str,
) -> tuple[Mapping[str, JsonValue], ...]:
    if type(value) is not tuple:
        _invalid_type(path, "a tuple of JSON objects")
    return tuple(
        _freeze_json_object(item, f"{path}[{index}]")
        for index, item in enumerate(cast(tuple[object, ...], value))
    )


def _mutable_json_mapping(
    value: Mapping[str, JsonValue],
) -> JsonObject:
    return {key: _mutable_json(item) for key, item in value.items()}


def _mutable_json(value: JsonValue) -> MutableJsonValue:
    if isinstance(value, Mapping):
        return {key: _mutable_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_json(item) for item in value]
    return value


def _json_mapping(value: object, path: str) -> Mapping[str, JsonValue]:
    return _freeze_json_object(_object(value, path), path)


def _json_records(
    value: object,
    path: str,
) -> tuple[Mapping[str, JsonValue], ...]:
    if type(value) is not list:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            path,
            "value must be a JSON array",
        )
    return tuple(
        _json_mapping(item, f"{path}[{index}]")
        for index, item in enumerate(cast(list[object], value))
    )


def _canonical_json(value: JsonObject) -> str:
    try:
        return dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (OverflowError, TypeError, UnicodeError, ValueError) as exc:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "continuation",
            "portable continuation contains a non-JSON value",
        ) from exc


def _continuation_text(value: object) -> str:
    if not isinstance(value, str | bytes):
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "continuation",
            "portable continuation must be text or UTF-8 bytes",
        )
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InputSnapshotError(
                InputErrorCode.SNAPSHOT_INVALID,
                "continuation",
                "portable continuation bytes must be UTF-8",
            ) from exc
    else:
        text = value
    if len(text.encode("utf-8")) > PORTABLE_CONTINUATION_MAX_UTF8_BYTES:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "continuation",
            "portable continuation exceeds its byte bound",
        )
    return text


def _unique_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object name")
        result[key] = value
    return result


def _object(value: object, path: str) -> dict[str, object]:
    if type(value) is not dict or any(
        type(key) is not str for key in cast(dict[object, object], value)
    ):
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            path,
            "value must be a JSON object with string keys",
        )
    return cast(dict[str, object], value)


def _require_fields(
    value: Mapping[str, object],
    expected: set[str],
    path: str,
) -> None:
    if set(value) != expected:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            path,
            "fields do not match the portable continuation schema",
        )


def _string(value: object, path: str) -> str:
    if type(value) is not str:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            path,
            "value must be text",
        )
    return value


def _integer(value: object, path: str) -> int:
    if type(value) is not int:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            path,
            "value must be an integer",
        )
    return value


def _datetime(value: object, path: str) -> datetime:
    text = _string(value, path)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            path,
            "value must be an ISO timestamp",
        ) from exc
    try:
        return validate_aware_datetime(parsed, path)
    except InputValidationError as exc:
        raise InputSnapshotError(exc.code, exc.path, exc.safe_message) from exc


def _validate_sha256(value: object, path: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            path,
            "value must be a lowercase SHA-256 digest",
        )
    return value


def _invalid_type(path: str, expected: str) -> None:
    raise InputValidationError(
        InputErrorCode.INVALID_TYPE,
        path,
        f"value must be {expected}",
    )


def _correlation_error(path: str, message: str) -> None:
    raise InputValidationError(
        InputErrorCode.CORRELATION_MISMATCH,
        path,
        message,
    )
