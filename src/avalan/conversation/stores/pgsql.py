"""Persist authenticated encrypted conversation checkpoints in PostgreSQL."""

from ...pgsql import (
    PgsqlCursor,
    PgsqlDatabase,
    PgsqlFailureCategory,
    PgsqlRow,
    PgsqlUnitOfWork,
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
    classify_pgsql_error,
)
from ..codec import (
    CHECKPOINT_CODEC_VERSION,
    ConversationCheckpointCodec,
    with_checkpoint_integrity,
)
from ..contract import (
    AuthorityScope,
    CheckpointId,
    CheckpointKind,
    IdempotencyDisposition,
    IdempotencyRecordState,
    LocalDeletionState,
    NamedHeadId,
    NamedHeadRevision,
    PortableContinuationReference,
    ProviderLaneId,
    PublicResponseId,
    RequestIdempotencyIdentity,
    UpstreamResponseId,
)
from ..crypto import (
    CONVERSATION_PAYLOAD_SCHEMA_VERSION,
    ConversationCipher,
    ConversationDataKey,
    ConversationKeyResolver,
    ConversationKeyStatus,
    ConversationPayloadAssociatedData,
    ConversationPayloadKind,
    EncryptedConversationPayload,
)
from ..durable_codec import (
    DURABLE_PAYLOAD_CODEC_VERSION,
    DurableConversationCodec,
    continuation_definition_digest,
    continuation_revision_binding_digest,
    execution_reservation_digest,
)
from ..errors import (
    ConversationAuthorizationError,
    ConversationConflictError,
    ConversationError,
    ConversationFeatureUnavailableError,
    ConversationKeyPolicyError,
    ConversationKeyRetiredError,
    ConversationLimitError,
    ConversationMigrationRequiredError,
    ConversationStorageError,
    ConversationTransitionError,
    ConversationValidationError,
)
from ..execution import (
    ConversationExecutionReservation,
    DurableToolRecoveryAdmission,
    DurableToolRecoveryLease,
    ProviderLaneExecutionAttestation,
    ProviderLaneExecutionStage,
    durable_tool_recovery_action,
    provider_lane_execution_receipt,
)
from ..lifecycle import (
    AmbiguousDispatchReconciliationDisposition,
    AmbiguousDispatchReconciliationRequest,
    AmbiguousDispatchReconciliationResult,
    AmbiguousDispatchResolution,
    LocalDeletionPreparation,
    ProviderLifecycleOrigin,
    ProviderLifecycleWorkRecord,
    ProviderLifecycleWorkState,
    ProviderQuarantineReceipt,
    ProviderQuarantineRequest,
)
from ..observability import authority_digest
from ..protocols import (
    ConversationClock,
    ConversationOutboxRecoveryWorker,
    ConversationUnitOfWork,
)
from ..runtime import (
    AtomicCommitReceipt,
    AtomicConversationCommit,
    CheckpointPage,
    IdempotencyResolution,
    IdempotencySettlementDisposition,
    IdempotencySettlementResolution,
    NamedHeadAdvance,
    OutboxClaimDisposition,
    OutboxClaimResolution,
    OutboxClaimTarget,
    OutboxRecord,
    OutboxRecoveryBatch,
    OutboxRecoveryDisposition,
    OutboxState,
    ProviderLaneOutputCandidate,
    ProvisionalPublicResponse,
    PruneReceipt,
    PublicationIntent,
    StoreCloseDisposition,
    StoreCloseResolution,
    StoreLimits,
    SweepReceipt,
)
from ..settings import (
    ConversationHandle,
    ConversationMode,
    ConversationResult,
    StatelessConversationHandle,
    StoredConversationHandle,
)
from ..state import (
    CheckpointCandidate,
    CheckpointLifecycle,
    ConversationCheckpoint,
    NamedHeadLifecycle,
    NamedHeadMetadata,
    NamedHeadSnapshot,
    StatelessProviderLaneSnapshot,
    StoredProviderLaneSnapshot,
    SuspensionCheckpointCandidate,
    SuspensionContinuationCheckpointCandidate,
    is_standalone_compact_bridge,
    validate_checkpoint_parent_kind,
)
from ..store import (
    InMemoryConversationStore,
    StoreAwaitBoundary,
    StoreBoundaryHook,
)
from ..value import (
    AuthorityDigest,
    ConversationCodecVersion,
    IntegrityDigest,
    validate_identifier,
)

from asyncio import CancelledError
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from importlib.util import find_spec
from itertools import count
from typing import Protocol, TypeVar, final
from uuid import uuid4

CONVERSATION_PGSQL_HEAD_REVISION = "20260801_0003"
CONVERSATION_PGSQL_APPLICATION_VERSION = 2
CONVERSATION_PGSQL_INSTALL_COMMAND = (
    'python3 -m pip install -U "avalan[task-pgsql,server]"'
)
CONVERSATION_PGSQL_MIGRATION_COMMAND = "avalan task pgsql migrate head"
_CHECKPOINT_ENVELOPE_LANE = ProviderLaneId("checkpoint-envelope")
_CONTINUATION_REFERENCE_LANE = ProviderLaneId("structured-input")
_CONCEALED_DIGEST = "0" * 64
_T = TypeVar("_T")


class PgsqlConversationFaultBoundary(StrEnum):
    """Identify every injectable PostgreSQL transaction boundary."""

    TRANSACTION_BEFORE = "transaction_before"
    SQL_BEFORE = "sql_before"
    SQL_AFTER = "sql_after"
    COMMIT_BEFORE = "commit_before"
    COMMIT_AFTER = "commit_after"
    ACKNOWLEDGEMENT_AFTER = "acknowledgement_after"
    OUTBOX_BEFORE = "outbox_before"
    OUTBOX_AFTER = "outbox_after"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PgsqlConversationFaultPoint:
    """Name one exact deterministic SQL or commit fault point."""

    boundary: PgsqlConversationFaultBoundary
    operation: str
    ordinal: int

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, PgsqlConversationFaultBoundary):
            raise ConversationValidationError()
        validate_identifier(self.operation, "operation")
        if type(self.ordinal) is not int or self.ordinal <= 0:
            raise ConversationValidationError()


class PgsqlConversationFaultHook(Protocol):
    """Inject deterministic failure around each durable SQL boundary."""

    async def reach(self, point: PgsqlConversationFaultPoint) -> None:
        """Reach one exact PostgreSQL fault point."""
        ...


@final
class _NoopPgsqlConversationFaultHook:
    async def reach(self, point: PgsqlConversationFaultPoint) -> None:
        if type(point) is not PgsqlConversationFaultPoint:
            raise ConversationValidationError()


@final
class _UtcConversationClock:
    async def now(self) -> datetime:
        return datetime.now(UTC)


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class PgsqlConversationStoreSettings:
    """Configure a bounded owned async PostgreSQL connection pool."""

    dsn: str
    schema: str | None = None
    pool_minimum: int = 1
    pool_maximum: int = 10
    pool_timeout_seconds: float = 10.0
    connect_timeout_seconds: int = 10
    statement_timeout_milliseconds: int = 30_000
    lock_timeout_milliseconds: int = 5_000
    idle_transaction_timeout_milliseconds: int = 30_000

    def __post_init__(self) -> None:
        validate_identifier(self.dsn, "dsn", max_length=8_192)
        if self.schema is not None:
            validate_identifier(self.schema, "schema", max_length=63)
        if (
            type(self.pool_minimum) is not int
            or self.pool_minimum <= 0
            or type(self.pool_maximum) is not int
            or self.pool_maximum < self.pool_minimum
            or self.pool_maximum > 64
        ):
            raise ConversationValidationError()
        if (
            not isinstance(self.pool_timeout_seconds, int | float)
            or isinstance(self.pool_timeout_seconds, bool)
            or self.pool_timeout_seconds <= 0
        ):
            raise ConversationValidationError()
        for value in (
            self.connect_timeout_seconds,
            self.statement_timeout_milliseconds,
            self.lock_timeout_milliseconds,
            self.idle_transaction_timeout_milliseconds,
        ):
            if type(value) is not int or value <= 0:
                raise ConversationValidationError()

    def database(self) -> PsycopgAsyncDatabase:
        """Return one closed bounded database wrapper."""
        return PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn=self.dsn,
                schema=self.schema,
                pool_minimum=self.pool_minimum,
                pool_maximum=self.pool_maximum,
                pool_timeout_seconds=self.pool_timeout_seconds,
                connect_timeout_seconds=self.connect_timeout_seconds,
                statement_timeout_milliseconds=(
                    self.statement_timeout_milliseconds
                ),
                lock_timeout_milliseconds=self.lock_timeout_milliseconds,
                idle_in_transaction_session_timeout_milliseconds=(
                    self.idle_transaction_timeout_milliseconds
                ),
                application_name="avalan-conversation-pgsql",
                autocommit=True,
                open=False,
            )
        )

    def __repr__(self) -> str:
        """Return pool settings without the PostgreSQL connection string."""
        return (
            "PgsqlConversationStoreSettings("
            "dsn=<redacted>, "
            f"schema={self.schema!r}, pool_minimum={self.pool_minimum}, "
            f"pool_maximum={self.pool_maximum})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PgsqlConversationStorePolicy:
    """Configure fail-closed durable schema and operation bounds."""

    limits: StoreLimits = StoreLimits()
    application_version: int = CONVERSATION_PGSQL_APPLICATION_VERSION
    minimum_schema_version: int = 1
    maximum_schema_version: int = 1
    max_batch_size: int = 100
    check_schema_on_open: bool = True
    test_only: bool = True

    def __post_init__(self) -> None:
        if type(self.limits) is not StoreLimits:
            raise ConversationValidationError()
        for value in (
            self.application_version,
            self.minimum_schema_version,
            self.maximum_schema_version,
            self.max_batch_size,
        ):
            if type(value) is not int or value <= 0:
                raise ConversationValidationError()
        if self.maximum_schema_version < self.minimum_schema_version:
            raise ConversationValidationError()
        if self.max_batch_size > self.limits.max_page_size:
            raise ConversationValidationError()
        if type(self.check_schema_on_open) is not bool:
            raise ConversationValidationError()
        if self.test_only is not True:
            raise ConversationFeatureUnavailableError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PgsqlConversationReadiness:
    """Report content-free durable migration and key-policy readiness."""

    schema_version: int
    minimum_reader_version: int
    maximum_reader_version: int
    minimum_writer_version: int
    maximum_writer_version: int
    checkpoint_codec_version: int
    application_version: int
    key_id: str
    key_revision: int

    def __post_init__(self) -> None:
        for value in (
            self.schema_version,
            self.minimum_reader_version,
            self.maximum_reader_version,
            self.minimum_writer_version,
            self.maximum_writer_version,
            self.checkpoint_codec_version,
            self.application_version,
            self.key_revision,
        ):
            if type(value) is not int or value <= 0:
                raise ConversationValidationError()
        validate_identifier(self.key_id, "key_id")


class ReconciliationWorkState(StrEnum):
    """Identify durable upstream deletion or key-rewrap work state."""

    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ReconciliationWorkRecord:
    """Carry one owner-leased reconciliation target with a redacted repr."""

    reconciliation_id: str
    checkpoint_id: CheckpointId
    lane_id: ProviderLaneId
    work_kind: str
    state: ReconciliationWorkState
    attempts: int
    upstream_response_id: UpstreamResponseId | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    binding_digest: IntegrityDigest | None = None
    checkpoint_lifecycle: CheckpointLifecycle | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.reconciliation_id, "reconciliation_id"),
            (self.checkpoint_id, "checkpoint_id"),
            (self.lane_id, "lane_id"),
            (self.work_kind, "work_kind"),
        ):
            validate_identifier(value, name)
        if self.work_kind not in {"delete_upstream", "rewrap_payload"}:
            raise ConversationValidationError()
        if (self.work_kind == "delete_upstream") != (
            self.upstream_response_id is not None
        ):
            raise ConversationValidationError()
        if self.upstream_response_id is not None:
            validate_identifier(
                self.upstream_response_id,
                "upstream_response_id",
            )
        if not isinstance(self.state, ReconciliationWorkState):
            raise ConversationValidationError()
        if type(self.attempts) is not int or self.attempts < 0:
            raise ConversationValidationError()
        claimed = self.state is ReconciliationWorkState.CLAIMED
        if claimed != (
            self.lease_owner is not None and self.lease_expires_at is not None
        ):
            raise ConversationValidationError()
        if self.lease_owner is not None:
            validate_identifier(self.lease_owner, "lease_owner")
        if self.lease_expires_at is not None:
            _validate_time(self.lease_expires_at)
        if self.binding_digest is not None:
            validate_identifier(self.binding_digest, "binding_digest")
        if self.checkpoint_lifecycle is not None and not isinstance(
            self.checkpoint_lifecycle,
            CheckpointLifecycle,
        ):
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return leased metadata without the private upstream target."""
        return (
            "ReconciliationWorkRecord("
            f"reconciliation_id={self.reconciliation_id!r}, "
            f"checkpoint_id={self.checkpoint_id!r}, "
            f"lane_id={self.lane_id!r}, work_kind={self.work_kind!r}, "
            f"state={self.state.value!r}, attempts={self.attempts}, "
            "upstream_response_id=<redacted>, "
            f"lease_owner={self.lease_owner!r}, "
            f"lease_expires_at={self.lease_expires_at!r}, "
            f"binding_digest={self.binding_digest!r}, "
            f"checkpoint_lifecycle={self.checkpoint_lifecycle!r})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class KeyRotationReceipt:
    """Report bounded content-free encrypted payload rotation counts."""

    examined: int
    reencrypted: int

    def __post_init__(self) -> None:
        if (
            type(self.examined) is not int
            or self.examined < 0
            or type(self.reencrypted) is not int
            or self.reencrypted < 0
            or self.reencrypted > self.examined
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class GarbageCollectionReceipt:
    """Report bounded unreferenced encrypted payload collection."""

    deleted_payloads: int

    def __post_init__(self) -> None:
        if type(self.deleted_payloads) is not int or self.deleted_payloads < 0:
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class _PreparedPayload:
    payload_id: str
    conversation_id: str
    associated_data: ConversationPayloadAssociatedData
    encrypted: EncryptedConversationPayload
    plaintext_bytes: int


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class _PreparedCheckpoint:
    checkpoint: ConversationCheckpoint
    key: ConversationDataKey
    envelope: _PreparedPayload
    outputs: tuple[_PreparedPayload, ...]
    deletion_targets: tuple[_PreparedPayload, ...]
    continuation: _PreparedPayload | None
    continuation_reference: PortableContinuationReference | None
    suspension_continuation: bool
    compact_continuation: bool

    @property
    def payloads(self) -> tuple[_PreparedPayload, ...]:
        """Return every encrypted payload in deterministic reference order."""
        continuation = (
            (self.continuation,) if self.continuation is not None else ()
        )
        return (
            (self.envelope,)
            + self.outputs
            + self.deletion_targets
            + continuation
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class _RotatedPayload:
    prepared: _PreparedPayload
    previous_key_id: str
    previous_key_revision: int
    previous_digest: str


@final
class PgsqlConversationUnitOfWork:
    """Own one validated durable candidate until commit or rollback."""

    def __init__(
        self,
        store: "PgsqlConversationStore",
        candidate: CheckpointCandidate,
    ) -> None:
        self._store = store
        self._candidate = candidate
        self._finished = False

    async def __aenter__(self) -> ConversationUnitOfWork:
        if self._finished:
            raise ConversationStorageError()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        if not self._finished:
            await self.rollback()

    async def commit(self) -> ConversationCheckpoint:
        if self._finished:
            raise ConversationStorageError()
        checkpoint = await self._store.commit(self._candidate)
        self._finished = True
        return checkpoint

    @property
    def database(self) -> PgsqlDatabase:
        """Return the database that must own the shared transaction."""
        return self._store.database

    @property
    def checkpoint_id(self) -> str:
        """Return the staged checkpoint identity."""
        return str(self._candidate.checkpoint.identity.checkpoint_id)

    @property
    def execution_segment_id(self) -> str:
        """Return the staged execution-segment identity."""
        return str(self._candidate.checkpoint.identity.execution_segment_id)

    @property
    def continuation_id(self) -> str | None:
        """Return the staged continuation identity when suspended."""
        if not isinstance(self._candidate, SuspensionCheckpointCandidate):
            return None
        return str(self._candidate.continuation.continuation_id)

    @property
    def continuation_state_revision(self) -> int | None:
        """Return the staged continuation revision when suspended."""
        if not isinstance(self._candidate, SuspensionCheckpointCandidate):
            return None
        return int(self._candidate.continuation.state_revision)

    async def commit_in(
        self,
        unit: PgsqlUnitOfWork,
    ) -> ConversationCheckpoint:
        """Insert the candidate inside an existing PostgreSQL transaction."""
        if self._finished:
            raise ConversationStorageError()
        return await self._store.commit_in_unit(unit, self._candidate)

    def settle_committed(self) -> None:
        """Mark the participant finished after its owner commits."""
        if self._finished:
            raise ConversationStorageError()
        self._finished = True

    async def rollback(self) -> None:
        self._finished = True


@final
class PgsqlConversationStore:
    """Persist test-only durable lanes against the Phase 3 store protocol."""

    @property
    def durable(self) -> bool:
        """Return true because committed state survives process restart."""
        return True

    def __init__(
        self,
        database: PgsqlDatabase,
        *,
        key_resolver: ConversationKeyResolver,
        cipher: ConversationCipher,
        policy: PgsqlConversationStorePolicy = PgsqlConversationStorePolicy(),
        checkpoint_codec: ConversationCheckpointCodec = (
            ConversationCheckpointCodec()
        ),
        durable_codec: DurableConversationCodec = DurableConversationCodec(),
        clock: ConversationClock | None = None,
        boundary_hook: StoreBoundaryHook | None = None,
        fault_hook: PgsqlConversationFaultHook | None = None,
        owns_database: bool = False,
    ) -> None:
        if not hasattr(database, "connection"):
            raise ConversationValidationError()
        if not hasattr(key_resolver, "current_write_key") or not hasattr(
            key_resolver, "read_key"
        ):
            raise ConversationValidationError()
        if not hasattr(cipher, "encrypt") or not hasattr(cipher, "decrypt"):
            raise ConversationValidationError()
        if (
            type(policy) is not PgsqlConversationStorePolicy
            or type(checkpoint_codec) is not ConversationCheckpointCodec
            or type(durable_codec) is not DurableConversationCodec
            or type(owns_database) is not bool
        ):
            raise ConversationValidationError()
        self._database = database
        self._key_resolver = key_resolver
        self._cipher = cipher
        self._policy = policy
        self._checkpoint_codec = checkpoint_codec
        self._durable_codec = durable_codec
        self._clock = clock or _UtcConversationClock()
        self._boundary_hook = boundary_hook
        self._fault_hook = fault_hook or _NoopPgsqlConversationFaultHook()
        self._owns_database = owns_database
        self._fault_ordinals = count(1)
        self._opened = False
        self._closed = False

    @classmethod
    def from_settings(
        cls,
        settings: PgsqlConversationStoreSettings,
        *,
        key_resolver: ConversationKeyResolver,
        cipher: ConversationCipher,
        policy: PgsqlConversationStorePolicy = PgsqlConversationStorePolicy(),
        clock: ConversationClock | None = None,
        boundary_hook: StoreBoundaryHook | None = None,
        fault_hook: PgsqlConversationFaultHook | None = None,
    ) -> "PgsqlConversationStore":
        """Create one store with an owned bounded async connection pool."""
        if type(settings) is not PgsqlConversationStoreSettings:
            raise ConversationValidationError()
        return cls(
            settings.database(),
            key_resolver=key_resolver,
            cipher=cipher,
            policy=policy,
            clock=clock,
            boundary_hook=boundary_hook,
            fault_hook=fault_hook,
            owns_database=True,
        )

    @property
    def database(self) -> PgsqlDatabase:
        """Return the database for explicit shared atomic units of work."""
        return self._database

    async def open(self) -> None:
        """Open owned resources and fail closed on migration drift."""
        self._ensure_not_closed()
        if self._opened:
            return
        if find_spec("psycopg") is None or find_spec("psycopg_pool") is None:
            raise ConversationFeatureUnavailableError()
        open_database = getattr(self._database, "open", None)
        if open_database is not None:
            result = open_database()
            if isinstance(result, Awaitable):
                await result
        if self._policy.check_schema_on_open:
            await self._schema_readiness()
        self._opened = True

    async def readiness(
        self,
        authority: AuthorityScope,
    ) -> PgsqlConversationReadiness:
        """Verify migration, application window, and current write key."""
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        self._ensure_open()
        metadata = await self._schema_readiness()
        digest = authority_digest(authority)
        key = await self._key_resolver.current_write_key(digest)
        if key.status is not ConversationKeyStatus.CURRENT:
            raise ConversationKeyPolicyError()
        return PgsqlConversationReadiness(
            schema_version=_row_int(metadata, "schema_version"),
            minimum_reader_version=_row_int(
                metadata, "minimum_reader_version"
            ),
            maximum_reader_version=_row_int(
                metadata, "maximum_reader_version"
            ),
            minimum_writer_version=_row_int(
                metadata, "minimum_writer_version"
            ),
            maximum_writer_version=_row_int(
                metadata, "maximum_writer_version"
            ),
            checkpoint_codec_version=_row_int(
                metadata, "checkpoint_codec_version"
            ),
            application_version=self._policy.application_version,
            key_id=key.key_id,
            key_revision=key.revision,
        )

    async def create(
        self,
        candidate: CheckpointCandidate,
    ) -> ConversationCheckpoint:
        await self._reach_store(StoreAwaitBoundary.CREATE)
        return await self._commit_candidate(candidate)

    async def create_with_named_head(
        self,
        candidate: CheckpointCandidate,
        advance: NamedHeadAdvance,
    ) -> ConversationCheckpoint:
        """Create a checkpoint and advance an exact head atomically."""
        if type(advance) is not NamedHeadAdvance:
            raise ConversationValidationError()
        await self._reach_store(StoreAwaitBoundary.CREATE)
        staged = InMemoryConversationStore._candidate_checkpoint(candidate)
        expected_head = NamedHeadMetadata(
            head_id=advance.head_id,
            revision=NamedHeadRevision(advance.expected_revision + 1),
        )
        parent_id = staged.identity.parent_checkpoint_id
        if parent_id is None or staged.head != expected_head:
            raise ConversationValidationError()
        parent = await self._load_checkpoint(parent_id, staged.authority)
        if (
            parent.kind is not CheckpointKind.STANDALONE_COMPACT_RESULT
            or parent.identity.parent_checkpoint_id
            != advance.parent_checkpoint_id
        ):
            raise ConversationValidationError()
        prepared = await self._prepare_checkpoint(
            candidate,
            committed_at=staged.timestamps.created_at,
            output_candidates=(),
        )
        authority_key = str(authority_digest(staged.authority))

        async def operation(cursor: PgsqlCursor) -> None:
            row = await self._fetchone(
                cursor,
                "compact_head_lock",
                _SELECT_HEAD_FOR_UPDATE_SQL,
                (authority_key, advance.head_id),
            )
            if (
                row is None
                or _row_str(row, "lifecycle_state") != "active"
                or _row_int(row, "head_revision") != advance.expected_revision
                or _row_str(row, "checkpoint_id")
                != str(advance.parent_checkpoint_id)
            ):
                raise ConversationConflictError()
            await self._insert_checkpoint(cursor, prepared)
            await self._execute(
                cursor,
                "compact_head_advance",
                _UPDATE_HEAD_SQL,
                (
                    prepared.checkpoint.identity.checkpoint_id,
                    prepared.checkpoint.timestamps.committed_at,
                    authority_key,
                    advance.head_id,
                    advance.expected_revision,
                    advance.parent_checkpoint_id,
                ),
            )

        await self._transaction("compact_checkpoint_head_commit", operation)
        return prepared.checkpoint

    async def load(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> ConversationCheckpoint:
        await self._reach_store(StoreAwaitBoundary.LOAD)
        return await self._load_checkpoint(checkpoint_id, authority)

    async def authorize(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> ConversationCheckpoint:
        await self._reach_store(StoreAwaitBoundary.AUTHORIZE)
        return await self._load_checkpoint(checkpoint_id, authority)

    async def stage(
        self,
        candidate: CheckpointCandidate,
    ) -> PgsqlConversationUnitOfWork:
        await self._reach_store(StoreAwaitBoundary.STAGE)
        InMemoryConversationStore._candidate_checkpoint(candidate)
        self._ensure_open()
        return PgsqlConversationUnitOfWork(self, candidate)

    async def commit(
        self,
        candidate: CheckpointCandidate,
    ) -> ConversationCheckpoint:
        await self._reach_store(StoreAwaitBoundary.COMMIT)
        return await self._commit_candidate(candidate)

    async def _commit_candidate(
        self,
        candidate: CheckpointCandidate,
    ) -> ConversationCheckpoint:
        staged = InMemoryConversationStore._candidate_checkpoint(candidate)
        prepared = await self._prepare_checkpoint(
            candidate,
            committed_at=staged.timestamps.created_at,
            output_candidates=(),
        )

        async def operation(cursor: PgsqlCursor) -> None:
            await self._insert_checkpoint(cursor, prepared)

        await self._transaction("checkpoint_create", operation)
        return prepared.checkpoint

    async def commit_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        candidate: CheckpointCandidate,
    ) -> ConversationCheckpoint:
        """Commit one checkpoint through a caller-owned transaction."""
        if type(unit) is not PgsqlUnitOfWork:
            raise ConversationValidationError()
        self._ensure_open()
        staged = InMemoryConversationStore._candidate_checkpoint(candidate)
        prepared = await self._prepare_checkpoint(
            candidate,
            committed_at=staged.timestamps.created_at,
            output_candidates=(),
        )
        await self._insert_checkpoint(unit.cursor, prepared)
        return prepared.checkpoint

    async def _prepare_checkpoint(
        self,
        candidate: CheckpointCandidate,
        *,
        committed_at: datetime,
        output_candidates: tuple[ProviderLaneOutputCandidate, ...],
        compact_continuation: bool = False,
    ) -> _PreparedCheckpoint:
        staged = InMemoryConversationStore._candidate_checkpoint(candidate)
        committed = InMemoryConversationStore._committed_checkpoint(
            staged, committed_at
        )
        encoded = self._checkpoint_codec.encode(committed)
        self._validate_checkpoint_limits(committed, len(encoded))
        digest = authority_digest(committed.authority)
        key = await self._key_resolver.current_write_key(digest)
        if key.status is not ConversationKeyStatus.CURRENT:
            raise ConversationKeyPolicyError()
        envelope = await self._encrypt_payload(
            encoded,
            key=key,
            authority=digest,
            conversation_id=str(committed.identity.conversation_id),
            checkpoint_id=committed.identity.checkpoint_id,
            lane_id=_CHECKPOINT_ENVELOPE_LANE,
            sequence=0,
            kind=ConversationPayloadKind.CHECKPOINT,
            codec_version=CHECKPOINT_CODEC_VERSION,
        )
        outputs: list[_PreparedPayload] = []
        for sequence, output in enumerate(output_candidates):
            outputs.append(
                await self._encrypt_payload(
                    self._durable_codec.encode_output(output),
                    key=key,
                    authority=digest,
                    conversation_id=str(committed.identity.conversation_id),
                    checkpoint_id=committed.identity.checkpoint_id,
                    lane_id=output.lane_id,
                    sequence=sequence,
                    kind=ConversationPayloadKind.LANE_OUTPUT,
                    codec_version=ConversationCodecVersion(
                        DURABLE_PAYLOAD_CODEC_VERSION
                    ),
                )
            )
        deletion_targets: list[_PreparedPayload] = []
        for lane in committed.content.lanes:
            if isinstance(lane, StoredProviderLaneSnapshot):
                deletion_targets.append(
                    await self._encrypt_payload(
                        str(lane.upstream_response_id).encode("utf-8"),
                        key=key,
                        authority=digest,
                        conversation_id=str(
                            committed.identity.conversation_id
                        ),
                        checkpoint_id=committed.identity.checkpoint_id,
                        lane_id=lane.lane_id,
                        sequence=0,
                        kind=ConversationPayloadKind.DELETION_TARGET,
                        codec_version=ConversationCodecVersion(
                            DURABLE_PAYLOAD_CODEC_VERSION
                        ),
                    )
                )
        continuation: _PreparedPayload | None = None
        continuation_reference: PortableContinuationReference | None = None
        if isinstance(candidate, SuspensionCheckpointCandidate):
            continuation_reference = candidate.continuation
            continuation = await self._encrypt_payload(
                self._durable_codec.encode_continuation_reference(
                    candidate.continuation
                ),
                key=key,
                authority=digest,
                conversation_id=str(committed.identity.conversation_id),
                checkpoint_id=committed.identity.checkpoint_id,
                lane_id=_CONTINUATION_REFERENCE_LANE,
                sequence=0,
                kind=ConversationPayloadKind.CONTINUATION_REFERENCE,
                codec_version=ConversationCodecVersion(
                    DURABLE_PAYLOAD_CODEC_VERSION
                ),
            )
        return _PreparedCheckpoint(
            checkpoint=committed,
            key=key,
            envelope=envelope,
            outputs=tuple(outputs),
            deletion_targets=tuple(deletion_targets),
            continuation=continuation,
            continuation_reference=continuation_reference,
            suspension_continuation=isinstance(
                candidate,
                SuspensionContinuationCheckpointCandidate,
            ),
            compact_continuation=compact_continuation,
        )

    async def _encrypt_payload(
        self,
        plaintext: bytes,
        *,
        key: ConversationDataKey,
        authority: AuthorityDigest,
        conversation_id: str,
        checkpoint_id: CheckpointId,
        lane_id: ProviderLaneId,
        sequence: int,
        kind: ConversationPayloadKind,
        codec_version: ConversationCodecVersion,
    ) -> _PreparedPayload:
        associated_data = ConversationPayloadAssociatedData(
            authority_digest=authority,
            checkpoint_id=checkpoint_id,
            lane_id=lane_id,
            sequence=sequence,
            payload_kind=kind,
            payload_schema_version=CONVERSATION_PAYLOAD_SCHEMA_VERSION,
            codec_version=codec_version,
            key_id=key.key_id,
            key_revision=key.revision,
        )
        encrypted = await self._cipher.encrypt(
            plaintext,
            key=key,
            associated_data=associated_data,
        )
        return _PreparedPayload(
            payload_id=f"conversation-payload-{uuid4().hex}",
            conversation_id=conversation_id,
            associated_data=associated_data,
            encrypted=encrypted,
            plaintext_bytes=len(plaintext),
        )

    async def _insert_checkpoint(
        self,
        cursor: PgsqlCursor,
        prepared: _PreparedCheckpoint,
        *,
        enforce_capacity: bool = True,
    ) -> None:
        checkpoint = prepared.checkpoint
        identity = checkpoint.identity
        authority_key = str(authority_digest(checkpoint.authority))
        if enforce_capacity and str(identity.checkpoint_id).startswith(
            "quarantine-"
        ):
            raise ConversationValidationError()
        await self._synchronize_write_key(cursor, authority_key, prepared.key)
        await self._execute(
            cursor,
            "checkpoint_create_conversation",
            _INSERT_CONVERSATION_SQL,
            (identity.conversation_id, authority_key),
        )
        conversation = await self._fetchone(
            cursor,
            "checkpoint_lock_conversation",
            _SELECT_CONVERSATION_FOR_UPDATE_SQL,
            (identity.conversation_id,),
        )
        if (
            conversation is None
            or _row_str(conversation, "authority_digest") != authority_key
            or _row_str(conversation, "lifecycle_state") != "active"
        ):
            raise ConversationAuthorizationError()
        if enforce_capacity:
            await self._validate_checkpoint_capacity(
                cursor,
                checkpoint,
                suspension_continuation=prepared.suspension_continuation,
                compact_continuation=prepared.compact_continuation,
            )
        counts = checkpoint.content.safe_counts
        total_payload_bytes = sum(
            value.plaintext_bytes for value in prepared.payloads
        )
        await self._execute(
            cursor,
            "checkpoint_insert",
            _INSERT_CHECKPOINT_SQL,
            (
                identity.checkpoint_id,
                identity.conversation_id,
                identity.logical_turn_id,
                identity.execution_segment_id,
                identity.branch_id,
                identity.parent_checkpoint_id,
                identity.sequence,
                identity.parent_sequence,
                checkpoint.kind.value,
                checkpoint.lifecycle.value,
                authority_key,
                int(CHECKPOINT_CODEC_VERSION),
                CONVERSATION_PAYLOAD_SCHEMA_VERSION,
                len(prepared.payloads),
                total_payload_bytes,
                counts.lane_count,
                counts.provider_item_count,
                counts.opaque_byte_count,
                checkpoint.timestamps.created_at,
                checkpoint.timestamps.committed_at,
                checkpoint.timestamps.expires_at,
            ),
        )
        for lane_sequence, lane in enumerate(checkpoint.content.lanes):
            receipt = lane.execution_receipt
            item_count = (
                lane.ledger.item_count
                if isinstance(lane, StatelessProviderLaneSnapshot)
                else 0
            )
            opaque_bytes = (
                sum(
                    item.opaque_state.byte_count
                    for item in lane.ledger.items
                    if item.opaque_state is not None
                )
                if isinstance(lane, StatelessProviderLaneSnapshot)
                else 0
            )
            await self._execute(
                cursor,
                "checkpoint_insert_lane",
                _INSERT_LANE_SQL,
                (
                    identity.checkpoint_id,
                    lane.lane_id,
                    lane_sequence,
                    (
                        ConversationMode.STATELESS.value
                        if isinstance(lane, StatelessProviderLaneSnapshot)
                        else ConversationMode.STORED.value
                    ),
                    str(lane.binding.integrity_digest),
                    str(receipt.digest) if receipt is not None else None,
                    item_count,
                    opaque_bytes,
                    (
                        "not_applicable"
                        if isinstance(lane, StatelessProviderLaneSnapshot)
                        else "pending"
                    ),
                ),
            )
        for payload in prepared.payloads:
            await self._insert_payload(cursor, payload)
        await self._insert_continuation_reference(cursor, prepared)

    async def _validate_checkpoint_capacity(
        self,
        cursor: PgsqlCursor,
        checkpoint: ConversationCheckpoint,
        *,
        suspension_continuation: bool = False,
        compact_continuation: bool = False,
    ) -> None:
        await self._lock_global_capacity(cursor)
        total = await self._fetchone(
            cursor,
            "checkpoint_count",
            _COUNT_CHECKPOINTS_SQL,
            None,
        )
        if total is None or _row_int(total, "record_count") >= (
            self._policy.limits.max_checkpoints
        ):
            raise ConversationLimitError()
        parent_id = checkpoint.identity.parent_checkpoint_id
        if parent_id is None:
            validate_checkpoint_parent_kind(
                checkpoint.kind,
                None,
                suspension_continuation=suspension_continuation,
                compact_continuation=compact_continuation,
            )
            return
        parent = await self._fetchone(
            cursor,
            "checkpoint_lock_parent",
            _SELECT_CHECKPOINT_FOR_UPDATE_SQL,
            (parent_id,),
        )
        expected_authority = str(authority_digest(checkpoint.authority))
        if (
            parent is None
            or _row_str(parent, "authority_digest") != expected_authority
            or _row_str(parent, "lifecycle_state") != "committed"
            or _row_str(parent, "conversation_id")
            != str(checkpoint.identity.conversation_id)
            or _row_int(parent, "checkpoint_sequence")
            != checkpoint.identity.parent_sequence
        ):
            raise ConversationAuthorizationError()
        validate_checkpoint_parent_kind(
            checkpoint.kind,
            CheckpointKind(_row_str(parent, "checkpoint_kind")),
            suspension_continuation=suspension_continuation,
            compact_continuation=compact_continuation,
        )
        children = await self._fetchone(
            cursor,
            "checkpoint_child_count",
            _COUNT_CHECKPOINT_CHILDREN_SQL,
            (parent_id,),
        )
        if children is None or _row_int(children, "record_count") >= (
            self._policy.limits.max_children_per_parent
        ):
            raise ConversationLimitError()

    async def _lock_global_capacity(self, cursor: PgsqlCursor) -> None:
        await self._execute(
            cursor,
            "global_capacity_lock",
            _LOCK_GLOBAL_CAPACITY_SQL,
            None,
        )

    async def _validate_outbox_capacity(self, cursor: PgsqlCursor) -> None:
        await self._lock_global_capacity(cursor)
        count_row = await self._fetchone(
            cursor,
            "outbox_count",
            _COUNT_OUTBOX_SQL,
            None,
        )
        if (
            count_row is None
            or _row_int(count_row, "record_count")
            >= self._policy.limits.max_outbox_records
        ):
            raise ConversationLimitError()

    async def _insert_payload(
        self,
        cursor: PgsqlCursor,
        prepared: _PreparedPayload,
    ) -> None:
        ad = prepared.associated_data
        encrypted = prepared.encrypted
        await self._execute(
            cursor,
            "payload_insert",
            _INSERT_PAYLOAD_SQL,
            (
                prepared.payload_id,
                ad.authority_digest,
                ad.checkpoint_id,
                prepared.conversation_id,
                ad.lane_id,
                ad.sequence,
                ad.payload_kind.value,
                ad.payload_schema_version,
                ad.codec_version,
                encrypted.key_id,
                encrypted.key_revision,
                encrypted.algorithm,
                encrypted.nonce,
                encrypted.ciphertext,
                encrypted.authenticated_digest,
            ),
        )
        await self._execute(
            cursor,
            "payload_reference_insert",
            _INSERT_PAYLOAD_REFERENCE_SQL,
            (
                ad.checkpoint_id,
                prepared.conversation_id,
                ad.authority_digest,
                ad.lane_id,
                ad.sequence,
                ad.payload_kind.value,
                ad.payload_schema_version,
                ad.codec_version,
                encrypted.key_id,
                encrypted.key_revision,
                encrypted.algorithm,
                encrypted.authenticated_digest,
                prepared.payload_id,
            ),
        )

    async def _insert_continuation_reference(
        self,
        cursor: PgsqlCursor,
        prepared: _PreparedCheckpoint,
    ) -> None:
        reference = prepared.continuation_reference
        payload = prepared.continuation
        if reference is None:
            if payload is not None:
                raise ConversationStorageError()
            return
        if payload is None:
            raise ConversationStorageError()
        checkpoint = prepared.checkpoint
        await self._execute(
            cursor,
            "continuation_reference_insert",
            _INSERT_CONTINUATION_REFERENCE_SQL,
            (
                checkpoint.identity.checkpoint_id,
                checkpoint.identity.conversation_id,
                authority_digest(checkpoint.authority),
                checkpoint.identity.execution_segment_id,
                reference.continuation_id,
                reference.state_revision,
                reference.digest,
                continuation_definition_digest(reference.definition),
                continuation_revision_binding_digest(
                    reference.revision_binding
                ),
                payload.associated_data.lane_id,
                payload.associated_data.sequence,
                payload.associated_data.payload_kind.value,
                payload.payload_id,
            ),
        )

    async def _synchronize_write_key(
        self,
        cursor: PgsqlCursor,
        authority_key: str,
        key: ConversationDataKey,
    ) -> None:
        await self._execute(
            cursor,
            "key_authority_register",
            _INSERT_KEY_AUTHORITY_SQL,
            (authority_key,),
        )
        authority = await self._fetchone(
            cursor,
            "key_authority_lock",
            _SELECT_KEY_AUTHORITY_FOR_UPDATE_SQL,
            (authority_key,),
        )
        if authority is None:
            raise ConversationStorageError()
        existing = await self._fetchone(
            cursor,
            "key_select_revision",
            _SELECT_KEY_REVISION_FOR_UPDATE_SQL,
            (authority_key, key.key_id, key.revision),
        )
        if (
            existing is not None
            and _row_str(existing, "key_status")
            == ConversationKeyStatus.RETIRED.value
        ):
            raise ConversationKeyRetiredError()
        generation_key = await self._fetchone(
            cursor,
            "key_select_generation",
            _SELECT_KEY_GENERATION_FOR_UPDATE_SQL,
            (authority_key, key.revision),
        )
        if (
            generation_key is not None
            and _row_str(generation_key, "key_id") != key.key_id
        ):
            raise ConversationKeyPolicyError()
        generation = _row_int(authority, "current_generation")
        if generation > key.revision:
            raise ConversationKeyPolicyError()
        if generation == key.revision and generation > 0:
            if (
                _row_optional_str(authority, "current_key_id") != key.key_id
                or _row_optional_int(authority, "current_key_revision")
                != key.revision
                or existing is None
                or _row_str(existing, "key_status")
                != ConversationKeyStatus.CURRENT.value
                or _row_str(existing, "algorithm") != key.algorithm
            ):
                raise ConversationKeyPolicyError()
            return
        if (
            existing is not None
            and _row_str(existing, "algorithm") != key.algorithm
        ):
            raise ConversationKeyPolicyError()
        await self._execute(
            cursor,
            "key_demote_current",
            _DEMOTE_CURRENT_KEYS_SQL,
            (authority_key, key.key_id, key.revision),
        )
        registered = await self._fetchone(
            cursor,
            "key_register_current",
            _UPSERT_CURRENT_KEY_SQL,
            (
                authority_key,
                key.key_id,
                key.revision,
                key.algorithm,
            ),
        )
        if registered is None:
            raise ConversationKeyRetiredError()
        cutover = await self._fetchone(
            cursor,
            "key_authority_cutover",
            _UPDATE_KEY_AUTHORITY_SQL,
            (
                key.revision,
                key.key_id,
                key.revision,
                authority_key,
                generation,
            ),
        )
        if cutover is None:
            raise ConversationConflictError()

    def _validate_checkpoint_limits(
        self,
        checkpoint: ConversationCheckpoint,
        encoded_bytes: int,
    ) -> None:
        limits = self._policy.limits
        counts = checkpoint.content.safe_counts
        if (
            encoded_bytes > limits.max_checkpoint_bytes
            or checkpoint.identity.sequence > limits.max_depth
            or counts.provider_item_count > limits.max_provider_items
        ):
            raise ConversationLimitError()

    async def reserve_idempotency(
        self,
        identity: RequestIdempotencyIdentity,
        *,
        execution: ConversationExecutionReservation | None = None,
    ) -> IdempotencyResolution:
        if type(identity) is not RequestIdempotencyIdentity:
            raise ConversationValidationError()
        if execution is not None and (
            type(execution) is not ConversationExecutionReservation
            or execution.idempotency != identity
        ):
            raise ConversationValidationError()
        await self._reach_store(StoreAwaitBoundary.IDEMPOTENCY)
        now = await self._clock.now()
        _validate_time(now)
        execution_digest_value = execution_reservation_digest(execution)
        authority_key = str(authority_digest(identity.authority))

        async def operation(cursor: PgsqlCursor) -> IdempotencyResolution:
            current = await self._fetchone(
                cursor,
                "idempotency_select",
                _SELECT_IDEMPOTENCY_FOR_UPDATE_SQL,
                (authority_key, identity.operation.value, identity.key),
            )
            if current is not None:
                if (
                    _row_str(current, "request_digest")
                    != str(identity.request_digest)
                    or _row_optional_str(current, "execution_digest")
                    != execution_digest_value
                ):
                    return IdempotencyResolution(
                        disposition=IdempotencyDisposition.CONFLICT
                    )
                state = IdempotencyRecordState(
                    _row_str(current, "record_state")
                )
                if state is IdempotencyRecordState.COMMITTED:
                    checkpoint_id = _row_optional_str(current, "checkpoint_id")
                    if checkpoint_id is None:
                        raise ConversationStorageError()
                    public_response_id = _row_optional_str(
                        current, "public_response_id"
                    )
                    return IdempotencyResolution(
                        disposition=IdempotencyDisposition.REPLAY_COMMITTED,
                        checkpoint_id=CheckpointId(checkpoint_id),
                        public_response_id=(
                            PublicResponseId(public_response_id)
                            if public_response_id is not None
                            else None
                        ),
                    )
                if state is IdempotencyRecordState.FAILED_NO_DISPATCH:
                    await self._cleanup_owner(
                        cursor,
                        _row_str(current, "owner_token"),
                    )
                    await self._execute(
                        cursor,
                        "idempotency_retry_delete",
                        _DELETE_IDEMPOTENCY_SQL,
                        (
                            authority_key,
                            identity.operation.value,
                            identity.key,
                        ),
                    )
                else:
                    if state is IdempotencyRecordState.IN_PROGRESS and (
                        _row_datetime(current, "lease_expires_at") <= now
                    ):
                        await self._execute(
                            cursor,
                            "idempotency_expire_fence",
                            _UPDATE_IDEMPOTENCY_STATE_SQL,
                            (
                                IdempotencyRecordState.AMBIGUOUS.value,
                                now,
                                authority_key,
                                identity.operation.value,
                                identity.key,
                            ),
                        )
                    return IdempotencyResolution(
                        disposition=IdempotencyDisposition.FENCED
                    )
            await self._lock_global_capacity(cursor)
            counts = await self._fetchone(
                cursor,
                "idempotency_count",
                _COUNT_IDEMPOTENCY_SQL,
                None,
            )
            in_flight = await self._fetchone(
                cursor,
                "idempotency_inflight_count",
                _COUNT_IDEMPOTENCY_IN_FLIGHT_SQL,
                None,
            )
            if (
                counts is None
                or in_flight is None
                or _row_int(counts, "record_count")
                >= self._policy.limits.max_idempotency_records
                or _row_int(in_flight, "record_count")
                >= self._policy.limits.max_in_flight
            ):
                raise ConversationLimitError()
            owner_token = f"reservation-owner-{uuid4().hex}"
            await self._execute(
                cursor,
                "idempotency_insert",
                _INSERT_IDEMPOTENCY_SQL,
                (
                    authority_key,
                    identity.operation.value,
                    identity.key,
                    identity.request_digest,
                    IdempotencyRecordState.IN_PROGRESS.value,
                    owner_token,
                    now
                    + timedelta(
                        seconds=self._policy.limits.idempotency_lease_seconds
                    ),
                    execution_digest_value,
                    now,
                    now,
                ),
            )
            if execution is not None:
                await self._insert_execution_reservation(cursor, execution)
            return IdempotencyResolution(
                disposition=IdempotencyDisposition.EXECUTE,
                owner_token=owner_token,
            )

        return await self._transaction("idempotency_reserve", operation)

    async def _insert_execution_reservation(
        self,
        cursor: PgsqlCursor,
        execution: ConversationExecutionReservation,
    ) -> None:
        identity = execution.identity
        authority_key = str(authority_digest(execution.idempotency.authority))
        for lane in execution.lanes:
            await self._execute(
                cursor,
                "execution_reservation_insert_lane",
                _INSERT_EXECUTION_RESERVATION_LANE_SQL,
                (
                    authority_key,
                    execution.idempotency.operation.value,
                    execution.idempotency.key,
                    identity.checkpoint_id,
                    identity.conversation_id,
                    identity.logical_turn_id,
                    identity.execution_segment_id,
                    identity.branch_id,
                    identity.sequence,
                    identity.parent_checkpoint_id,
                    identity.parent_sequence,
                    lane.binding.lane_id,
                    lane.binding.integrity_digest,
                    lane.mode.value,
                    lane.scope.value,
                ),
            )

    async def admit_tool_recovery(
        self,
        admission: DurableToolRecoveryAdmission,
        execution: ConversationExecutionReservation,
    ) -> DurableToolRecoveryLease:
        """Atomically lease one exact ambiguous durable tool suffix."""
        if (
            type(admission) is not DurableToolRecoveryAdmission
            or type(execution) is not ConversationExecutionReservation
            or execution.idempotency != admission.idempotency
        ):
            raise ConversationValidationError()
        checkpoint = await self.load(
            admission.checkpoint_id,
            admission.idempotency.authority,
        )
        _validate_tool_recovery_checkpoint(
            admission,
            execution,
            checkpoint,
        )
        await self._reach_store(StoreAwaitBoundary.IDEMPOTENCY)
        now = await self._clock.now()
        _validate_time(now)
        owner_token = f"recovery-owner-{uuid4().hex}"
        authority_key = str(authority_digest(admission.idempotency.authority))
        expected_execution_digest = execution_reservation_digest(execution)

        async def operation(cursor: PgsqlCursor) -> None:
            checkpoint_row = await self._fetchone(
                cursor,
                "tool_recovery_checkpoint_lock",
                _SELECT_CHECKPOINT_FOR_UPDATE_SQL,
                (admission.checkpoint_id,),
            )
            idempotency = await self._fetchone(
                cursor,
                "tool_recovery_idempotency_lock",
                _SELECT_IDEMPOTENCY_FOR_UPDATE_SQL,
                (
                    authority_key,
                    admission.idempotency.operation.value,
                    admission.idempotency.key,
                ),
            )
            if (
                checkpoint_row is None
                or _row_str(checkpoint_row, "authority_digest")
                != authority_key
                or _row_str(checkpoint_row, "lifecycle_state")
                != CheckpointLifecycle.COMMITTED.value
                or _row_str(checkpoint_row, "checkpoint_kind")
                != CheckpointKind.INTERNAL_PROVIDER_BOUNDARY.value
                or idempotency is None
                or _row_str(idempotency, "request_digest")
                != str(admission.idempotency.request_digest)
                or _row_str(idempotency, "record_state")
                != IdempotencyRecordState.AMBIGUOUS.value
                or _row_optional_str(idempotency, "execution_digest")
                != expected_execution_digest
            ):
                raise ConversationConflictError()
            await self._execute(
                cursor,
                "tool_recovery_admit",
                _UPDATE_IDEMPOTENCY_RECOVERY_LEASE_SQL,
                (
                    IdempotencyRecordState.IN_PROGRESS.value,
                    owner_token,
                    now
                    + timedelta(
                        seconds=self._policy.limits.idempotency_lease_seconds
                    ),
                    now,
                    authority_key,
                    admission.idempotency.operation.value,
                    admission.idempotency.key,
                ),
            )

        await self._transaction("tool_recovery_admit", operation)
        return DurableToolRecoveryLease(
            admission=admission,
            owner_token=owner_token,
        )

    async def stage_execution(
        self,
        stage: ProviderLaneExecutionStage,
    ) -> ProviderLaneExecutionAttestation:
        if type(stage) is not ProviderLaneExecutionStage:
            raise ConversationValidationError()
        expected_receipt = provider_lane_execution_receipt(
            authority=stage.idempotency.authority,
            identity=stage.identity,
            binding=stage.binding,
            mode=stage.mode,
            scope=stage.scope,
            completed_items=stage.completed_items,
            reasoning=stage.reasoning,
            usage=stage.usage,
            upstream_response_id=stage.upstream_response_id,
        )
        if stage.execution_receipt != expected_receipt:
            raise ConversationValidationError()
        await self._reach_store(StoreAwaitBoundary.EXECUTION_STAGE)
        authority_key = str(authority_digest(stage.idempotency.authority))

        async def operation(
            cursor: PgsqlCursor,
        ) -> ProviderLaneExecutionAttestation:
            reservation = await self._fetchone(
                cursor,
                "execution_stage_reservation",
                _SELECT_EXECUTION_RESERVATION_LANE_SQL,
                (
                    authority_key,
                    stage.idempotency.operation.value,
                    stage.idempotency.key,
                    stage.binding.lane_id,
                ),
            )
            idempotency = await self._fetchone(
                cursor,
                "execution_stage_idempotency",
                _SELECT_IDEMPOTENCY_FOR_UPDATE_SQL,
                (
                    authority_key,
                    stage.idempotency.operation.value,
                    stage.idempotency.key,
                ),
            )
            if (
                reservation is None
                or idempotency is None
                or _row_str(idempotency, "request_digest")
                != str(stage.idempotency.request_digest)
                or _row_str(idempotency, "owner_token") != stage.owner_token
                or _row_str(idempotency, "record_state")
                != IdempotencyRecordState.IN_PROGRESS.value
                or not _reservation_matches_stage(reservation, stage)
            ):
                raise ConversationConflictError()
            await self._lock_global_capacity(cursor)
            count_row = await self._fetchone(
                cursor,
                "execution_stage_count",
                _COUNT_EXECUTION_STAGING_SQL,
                None,
            )
            if (
                count_row is None
                or _row_int(count_row, "record_count")
                >= self._policy.limits.max_staged_execution_records
            ):
                raise ConversationLimitError()
            staging_id = f"execution-stage-{uuid4().hex}"
            await self._execute(
                cursor,
                "execution_stage_insert",
                _INSERT_EXECUTION_STAGING_SQL,
                (
                    staging_id,
                    authority_key,
                    stage.idempotency.operation.value,
                    stage.idempotency.key,
                    stage.idempotency.request_digest,
                    stage.owner_token,
                    stage.identity.checkpoint_id,
                    stage.binding.lane_id,
                    stage.binding.integrity_digest,
                    expected_receipt.digest,
                    stage.mode.value,
                    stage.scope.value,
                    expected_receipt.item_count,
                    expected_receipt.opaque_byte_count,
                ),
            )
            return ProviderLaneExecutionAttestation(
                schema_version=1,
                staging_id=staging_id,
                lane_id=stage.binding.lane_id,
            )

        return await self._transaction("execution_stage", operation)

    async def allocate_public_response(
        self,
        allocation: ProvisionalPublicResponse,
    ) -> None:
        if type(allocation) is not ProvisionalPublicResponse:
            raise ConversationValidationError()
        await self._reach_store(StoreAwaitBoundary.ALLOCATE)

        async def operation(cursor: PgsqlCursor) -> None:
            owner = await self._fetchone(
                cursor,
                "provisional_select_owner",
                _SELECT_IDEMPOTENCY_BY_OWNER_FOR_UPDATE_SQL,
                (allocation.owner_token,),
            )
            if (
                owner is None
                or _row_str(owner, "record_state")
                != IdempotencyRecordState.IN_PROGRESS.value
                or _row_str(owner, "authority_digest")
                != allocation.authority_digest
            ):
                raise ConversationConflictError()
            await self._lock_global_capacity(cursor)
            counts = await self._fetchone(
                cursor,
                "provisional_counts",
                _COUNT_RESPONSE_ALLOCATIONS_SQL,
                None,
            )
            if counts is None or (
                _row_int(counts, "provisional_count")
                >= self._policy.limits.max_provisional_responses
                or _row_int(counts, "total_count")
                >= self._policy.limits.max_public_responses
            ):
                raise ConversationLimitError()
            await self._execute(
                cursor,
                "provisional_insert",
                _INSERT_PROVISIONAL_SQL,
                (
                    allocation.provisional_response_id,
                    allocation.public_response_id,
                    allocation.owner_token,
                    allocation.authority_digest,
                ),
            )

        await self._transaction("provisional_allocate", operation)

    async def commit_atomic(
        self,
        commit: AtomicConversationCommit,
    ) -> AtomicCommitReceipt:
        if type(commit) is not AtomicConversationCommit:
            raise ConversationValidationError()
        InMemoryConversationStore._validate_atomic_commit_value(commit)
        await self._reach_store(StoreAwaitBoundary.COMMIT_ATOMIC)
        parent: ConversationCheckpoint | None = None
        parent_id = commit.candidate.checkpoint.identity.parent_checkpoint_id
        if parent_id is not None:
            parent = await self._load_checkpoint(
                parent_id,
                commit.candidate.checkpoint.authority,
            )
        compact_source = None
        if (
            parent is not None
            and parent.kind is CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
            and parent.identity.parent_checkpoint_id is not None
        ):
            compact_source = await self._load_checkpoint(
                parent.identity.parent_checkpoint_id,
                commit.candidate.checkpoint.authority,
            )
        prepared = await self._prepare_checkpoint(
            commit.candidate,
            committed_at=commit.committed_at,
            output_candidates=commit.output_candidates,
            compact_continuation=is_standalone_compact_bridge(
                parent,
                compact_source,
            ),
        )
        InMemoryConversationStore._validate_output_candidates(
            prepared.checkpoint,
            commit.output_candidates,
            parent=parent,
        )
        authority_key = str(authority_digest(prepared.checkpoint.authority))
        result = InMemoryConversationStore._build_result(
            commit, prepared.checkpoint
        )
        outbox = self._build_pending_outbox(
            commit,
            prepared.checkpoint,
            result,
        )

        async def operation(cursor: PgsqlCursor) -> None:
            await self._validate_atomic_reservation(
                cursor,
                commit,
                authority_key,
            )
            await self._validate_atomic_staging(cursor, commit)
            await self._validate_atomic_provisional(
                cursor,
                commit,
                authority_key,
            )
            await self._validate_atomic_head(cursor, commit)
            await self._insert_checkpoint(cursor, prepared)
            if outbox is not None:
                await self._validate_outbox_capacity(cursor)
            if commit.public_response_id is not None:
                await self._execute(
                    cursor,
                    "public_response_insert",
                    _INSERT_PUBLIC_RESPONSE_SQL,
                    (
                        commit.public_response_id,
                        prepared.checkpoint.identity.checkpoint_id,
                        authority_key,
                    ),
                )
                assert commit.provisional_response_id is not None
                await self._execute(
                    cursor,
                    "provisional_consume",
                    _DELETE_PROVISIONAL_SQL,
                    (commit.provisional_response_id,),
                )
            if commit.head_id is not None:
                assert commit.expected_head_revision is not None
                await self._execute(
                    cursor,
                    "head_advance",
                    _UPDATE_HEAD_SQL,
                    (
                        prepared.checkpoint.identity.checkpoint_id,
                        commit.committed_at,
                        authority_key,
                        commit.head_id,
                        commit.expected_head_revision,
                        prepared.checkpoint.identity.parent_checkpoint_id,
                    ),
                )
            if outbox is not None:
                await self._reach_fault(
                    PgsqlConversationFaultBoundary.OUTBOX_BEFORE,
                    "outbox_insert",
                )
                await self._execute(
                    cursor,
                    "outbox_insert",
                    _INSERT_OUTBOX_SQL,
                    (
                        outbox.intent.intent_id,
                        outbox.intent.checkpoint_id,
                        outbox.intent.public_response_id,
                        outbox.authority_digest,
                    ),
                )
                await self._reach_fault(
                    PgsqlConversationFaultBoundary.OUTBOX_AFTER,
                    "outbox_insert",
                )
            await self._execute(
                cursor,
                "execution_stage_consume",
                _DELETE_EXECUTION_STAGING_OWNER_SQL,
                (
                    commit.owner_token,
                    prepared.checkpoint.identity.checkpoint_id,
                ),
            )
            await self._execute(
                cursor,
                "idempotency_commit",
                _COMMIT_IDEMPOTENCY_SQL,
                (
                    prepared.checkpoint.identity.checkpoint_id,
                    commit.public_response_id,
                    commit.committed_at,
                    authority_key,
                    commit.idempotency.operation.value,
                    commit.idempotency.key,
                    commit.owner_token,
                    commit.idempotency.request_digest,
                ),
            )

        await self._transaction("checkpoint_atomic_commit", operation)
        await self._reach_fault(
            PgsqlConversationFaultBoundary.ACKNOWLEDGEMENT_AFTER,
            "checkpoint_atomic_commit",
        )
        return AtomicCommitReceipt(
            checkpoint=prepared.checkpoint,
            result=result,
            outbox=outbox,
            output_candidates=commit.output_candidates,
        )

    async def _validate_atomic_reservation(
        self,
        cursor: PgsqlCursor,
        commit: AtomicConversationCommit,
        authority_key: str,
    ) -> None:
        row = await self._fetchone(
            cursor,
            "atomic_idempotency_lock",
            _SELECT_IDEMPOTENCY_FOR_UPDATE_SQL,
            (
                authority_key,
                commit.idempotency.operation.value,
                commit.idempotency.key,
            ),
        )
        if (
            row is None
            or _row_str(row, "record_state")
            != IdempotencyRecordState.IN_PROGRESS.value
            or _row_str(row, "owner_token") != commit.owner_token
            or _row_str(row, "request_digest")
            != str(commit.idempotency.request_digest)
            or _row_optional_str(row, "execution_digest") is None
        ):
            raise ConversationConflictError()

    async def _validate_atomic_staging(
        self,
        cursor: PgsqlCursor,
        commit: AtomicConversationCommit,
    ) -> None:
        rows = await self._fetchall(
            cursor,
            "atomic_staging_lock",
            _SELECT_EXECUTION_STAGING_OWNER_SQL,
            (
                commit.owner_token,
                commit.candidate.checkpoint.identity.checkpoint_id,
            ),
        )
        attestations = {
            str(value.lane_id): value
            for value in commit.execution_attestations
        }
        candidates = {
            str(value.lane_id): value for value in commit.output_candidates
        }
        if len(rows) != len(candidates) or set(attestations) != set(
            candidates
        ):
            raise ConversationConflictError()
        by_lane = {_row_str(row, "lane_id"): row for row in rows}
        if set(by_lane) != set(candidates):
            raise ConversationConflictError()
        for lane_id, candidate in candidates.items():
            row = by_lane[lane_id]
            attestation = attestations[lane_id]
            if (
                _row_str(row, "staging_id") != attestation.staging_id
                or _row_str(row, "request_digest")
                != str(commit.idempotency.request_digest)
                or _row_str(row, "binding_digest")
                != str(candidate.binding.integrity_digest)
                or _row_str(row, "execution_digest")
                != str(candidate.execution_receipt.digest)
                or _row_str(row, "lane_mode") != candidate.mode.value
                or _row_str(row, "output_scope") != candidate.scope.value
                or _row_int(row, "item_count")
                != candidate.execution_receipt.item_count
                or _row_int(row, "opaque_byte_count")
                != candidate.execution_receipt.opaque_byte_count
            ):
                raise ConversationConflictError()

    async def _validate_atomic_provisional(
        self,
        cursor: PgsqlCursor,
        commit: AtomicConversationCommit,
        authority_key: str,
    ) -> None:
        if commit.provisional_response_id is None:
            return
        row = await self._fetchone(
            cursor,
            "atomic_provisional_lock",
            _SELECT_PROVISIONAL_FOR_UPDATE_SQL,
            (commit.provisional_response_id,),
        )
        if (
            row is None
            or _row_str(row, "owner_token") != commit.owner_token
            or _row_str(row, "authority_digest") != authority_key
            or _row_str(row, "public_response_id")
            != str(commit.public_response_id)
        ):
            raise ConversationConflictError()

    async def _validate_atomic_head(
        self,
        cursor: PgsqlCursor,
        commit: AtomicConversationCommit,
    ) -> None:
        if commit.head_id is None:
            return
        authority_key = str(authority_digest(commit.idempotency.authority))
        row = await self._fetchone(
            cursor,
            "atomic_head_lock",
            _SELECT_HEAD_FOR_UPDATE_SQL,
            (authority_key, commit.head_id),
        )
        if (
            row is None
            or _row_str(row, "lifecycle_state") != "active"
            or _row_int(row, "head_revision") != commit.expected_head_revision
            or _row_str(row, "checkpoint_id")
            != str(commit.candidate.checkpoint.identity.parent_checkpoint_id)
        ):
            raise ConversationConflictError()

    @staticmethod
    def _build_pending_outbox(
        commit: AtomicConversationCommit,
        checkpoint: ConversationCheckpoint,
        result: ConversationResult | None,
    ) -> OutboxRecord | None:
        if commit.outbox_intent_id is None:
            return None
        assert result is not None
        assert commit.public_response_id is not None
        return OutboxRecord(
            intent=PublicationIntent(
                intent_id=commit.outbox_intent_id,
                public_response_id=commit.public_response_id,
                checkpoint_id=checkpoint.identity.checkpoint_id,
                lane_outputs=result.lane_outputs,
            ),
            authority_digest=authority_digest(checkpoint.authority),
            state=OutboxState.PENDING,
        )

    async def create_head(
        self,
        head: NamedHeadSnapshot,
        authority: AuthorityScope,
    ) -> None:
        if type(head) is not NamedHeadSnapshot:
            raise ConversationValidationError()
        await self._reach_store(StoreAwaitBoundary.HEAD)
        authority_key = str(authority_digest(authority))

        async def operation(cursor: PgsqlCursor) -> None:
            checkpoint = await self._fetchone(
                cursor,
                "head_checkpoint_lock",
                _SELECT_CHECKPOINT_FOR_UPDATE_SQL,
                (head.checkpoint_id,),
            )
            await self._lock_global_capacity(cursor)
            count_row = await self._fetchone(
                cursor,
                "head_count",
                _COUNT_HEADS_SQL,
                None,
            )
            if (
                checkpoint is None
                or _row_str(checkpoint, "authority_digest") != authority_key
                or _row_str(checkpoint, "lifecycle_state") != "committed"
            ):
                raise ConversationAuthorizationError()
            if (
                count_row is None
                or _row_int(count_row, "record_count")
                >= self._policy.limits.max_heads
            ):
                raise ConversationLimitError()
            await self._execute(
                cursor,
                "head_insert",
                _INSERT_HEAD_SQL,
                (
                    authority_key,
                    head.head_id,
                    head.revision,
                    head.checkpoint_id,
                    head.lifecycle.value,
                ),
            )

        await self._transaction("head_create", operation)

    async def load_head(
        self,
        head_id: NamedHeadId,
        authority: AuthorityScope,
    ) -> NamedHeadSnapshot:
        validate_identifier(head_id, "head_id")
        await self._reach_store(StoreAwaitBoundary.HEAD)
        authority_key = str(authority_digest(authority))
        row = await self._read_one(
            "head_load",
            _SELECT_HEAD_SQL,
            (authority_key, head_id),
        )
        if row is None or _row_str(row, "lifecycle_state") != "active":
            raise ConversationAuthorizationError()
        return NamedHeadSnapshot(
            head_id=head_id,
            revision=NamedHeadRevision(_row_int(row, "head_revision")),
            checkpoint_id=CheckpointId(_row_str(row, "checkpoint_id")),
            lifecycle=NamedHeadLifecycle.ACTIVE,
        )

    async def branch_count(
        self,
        parent_checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> int:
        await self._reach_store(StoreAwaitBoundary.BRANCH)
        authority_key = str(authority_digest(authority))
        row = await self._read_one(
            "branch_count",
            _COUNT_AUTHORIZED_CHILDREN_SQL,
            (parent_checkpoint_id, authority_key, parent_checkpoint_id),
        )
        if row is None or _row_int(row, "parent_count") != 1:
            raise ConversationAuthorizationError()
        return _row_int(row, "child_count")

    async def _load_checkpoint(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> ConversationCheckpoint:
        return await self._load_checkpoint_lifecycle(
            checkpoint_id,
            authority,
            CheckpointLifecycle.COMMITTED,
        )

    async def _load_checkpoint_lifecycle(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
        lifecycle: CheckpointLifecycle,
    ) -> ConversationCheckpoint:
        validate_identifier(checkpoint_id, "checkpoint_id")
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        if not isinstance(lifecycle, CheckpointLifecycle):
            raise ConversationValidationError()
        self._ensure_open()
        authority_key = authority_digest(authority)
        row = await self._read_one(
            "checkpoint_load",
            _SELECT_CHECKPOINT_PAYLOAD_SQL,
            (checkpoint_id, authority_key),
        )
        if row is None or _row_str(row, "lifecycle_state") != lifecycle.value:
            raise ConversationAuthorizationError()
        self._validate_payload_reference_row(
            row,
            checkpoint_id=checkpoint_id,
            authority=authority_key,
            kind=ConversationPayloadKind.CHECKPOINT,
            lane_id=_CHECKPOINT_ENVELOPE_LANE,
            sequence=0,
        )
        plaintext = await self._decrypt_payload_row(row)
        checkpoint = self._checkpoint_codec.decode(plaintext)
        if (
            checkpoint.identity.checkpoint_id != checkpoint_id
            or checkpoint.authority != authority
            or str(authority_digest(checkpoint.authority))
            != str(authority_key)
            or checkpoint.lifecycle is not lifecycle
            or checkpoint.identity.execution_segment_id
            != _row_str(row, "execution_segment_id")
            or checkpoint.identity.sequence
            != _row_int(row, "checkpoint_sequence")
            or checkpoint.integrity is None
        ):
            raise ConversationStorageError()
        return checkpoint

    @staticmethod
    def _validate_payload_reference_row(
        row: PgsqlRow,
        *,
        checkpoint_id: CheckpointId,
        authority: AuthorityDigest,
        kind: ConversationPayloadKind,
        lane_id: ProviderLaneId,
        sequence: int,
    ) -> None:
        payload_id = _row_str(row, "payload_id")
        conversation_id = _row_str(row, "conversation_id")
        authority_key = str(authority)
        payload_kind = _row_str(row, "payload_kind")
        payload_lane = _row_str(row, "lane_id")
        payload_sequence = _row_int(row, "payload_sequence")
        payload_schema = _row_int(row, "payload_schema_version")
        codec_version = _row_int(row, "codec_version")
        key_id = _row_str(row, "key_id")
        key_revision = _row_int(row, "key_revision")
        algorithm = _row_str(row, "algorithm")
        authenticated_digest = _row_str(row, "authenticated_digest")
        if (
            _row_str(row, "checkpoint_id") != str(checkpoint_id)
            or _row_str(row, "authority_digest") != authority_key
            or payload_kind != kind.value
            or payload_lane != str(lane_id)
            or payload_sequence != sequence
            or _row_str(row, "checkpoint_conversation_id") != conversation_id
            or _row_str(row, "checkpoint_authority_digest") != authority_key
            or _row_str(row, "reference_checkpoint_id") != str(checkpoint_id)
            or _row_str(row, "reference_conversation_id") != conversation_id
            or _row_str(row, "reference_authority_digest") != authority_key
            or _row_str(row, "reference_lane_id") != payload_lane
            or _row_int(row, "reference_payload_sequence") != payload_sequence
            or _row_str(row, "reference_payload_kind") != payload_kind
            or _row_int(row, "reference_payload_schema_version")
            != payload_schema
            or _row_int(row, "reference_codec_version") != codec_version
            or _row_str(row, "reference_key_id") != key_id
            or _row_int(row, "reference_key_revision") != key_revision
            or _row_str(row, "reference_algorithm") != algorithm
            or _row_str(row, "reference_authenticated_digest")
            != authenticated_digest
            or _row_str(row, "reference_payload_id") != payload_id
            or _row_int(row, "checkpoint_payload_schema_version")
            != payload_schema
            or (
                kind is ConversationPayloadKind.CHECKPOINT
                and _row_int(row, "checkpoint_codec_version") != codec_version
            )
        ):
            raise ConversationStorageError()

    async def _decrypt_payload_row(self, row: PgsqlRow) -> bytes:
        authority_key = AuthorityDigest(_row_str(row, "authority_digest"))
        key_id = _row_str(row, "key_id")
        key_revision = _row_int(row, "key_revision")
        if _row_str(row, "key_status") == ConversationKeyStatus.RETIRED.value:
            raise ConversationKeyRetiredError()
        key = await self._key_resolver.read_key(
            authority_key,
            key_id=key_id,
            revision=key_revision,
        )
        associated_data = ConversationPayloadAssociatedData(
            authority_digest=authority_key,
            checkpoint_id=CheckpointId(_row_str(row, "checkpoint_id")),
            lane_id=ProviderLaneId(_row_str(row, "lane_id")),
            sequence=_row_int(row, "payload_sequence"),
            payload_kind=ConversationPayloadKind(
                _row_str(row, "payload_kind")
            ),
            payload_schema_version=_row_int(row, "payload_schema_version"),
            codec_version=ConversationCodecVersion(
                _row_int(row, "codec_version")
            ),
            key_id=key_id,
            key_revision=key_revision,
        )
        payload = EncryptedConversationPayload(
            nonce=_row_bytes(row, "nonce"),
            ciphertext=_row_bytes(row, "ciphertext"),
            authenticated_digest=_row_str(row, "authenticated_digest"),
            associated_data_digest=sha256(
                associated_data.encode()
            ).hexdigest(),
            key_id=key_id,
            key_revision=key_revision,
            algorithm=_row_str(row, "algorithm"),
        )
        return await self._cipher.decrypt(
            payload,
            key=key,
            associated_data=associated_data,
        )

    async def retrieve_output_candidates(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> tuple[ProviderLaneOutputCandidate, ...]:
        validate_identifier(checkpoint_id, "checkpoint_id")
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        await self._reach_store(StoreAwaitBoundary.RETRIEVE_OUTPUTS)
        rows = await self._read_all(
            "outputs_load",
            _SELECT_OUTPUT_PAYLOADS_SQL,
            (checkpoint_id, authority_digest(authority)),
        )
        if not rows:
            raise ConversationAuthorizationError()
        candidates: list[ProviderLaneOutputCandidate] = []
        for expected_sequence, row in enumerate(rows):
            lane_id = ProviderLaneId(_row_str(row, "lane_id"))
            self._validate_payload_reference_row(
                row,
                checkpoint_id=checkpoint_id,
                authority=authority_digest(authority),
                kind=ConversationPayloadKind.LANE_OUTPUT,
                lane_id=lane_id,
                sequence=expected_sequence,
            )
            if _row_str(row, "registered_lane_id") != str(lane_id):
                raise ConversationStorageError()
            decoded = self._durable_codec.decode_output(
                await self._decrypt_payload_row(row)
            )
            if decoded.lane_id != lane_id:
                raise ConversationStorageError()
            candidates.append(decoded)
        return tuple(candidates)

    async def load_continuation_reference(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> PortableContinuationReference:
        """Load one exact encrypted structured-input checkpoint reference."""
        validate_identifier(checkpoint_id, "checkpoint_id")
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        row = await self._read_one(
            "continuation_reference_load",
            _SELECT_CONTINUATION_PAYLOAD_SQL,
            (checkpoint_id, authority_digest(authority)),
        )
        if row is None:
            raise ConversationAuthorizationError()
        self._validate_payload_reference_row(
            row,
            checkpoint_id=checkpoint_id,
            authority=authority_digest(authority),
            kind=ConversationPayloadKind.CONTINUATION_REFERENCE,
            lane_id=_CONTINUATION_REFERENCE_LANE,
            sequence=0,
        )
        if (
            _row_str(row, "continuation_conversation_id")
            != _row_str(row, "conversation_id")
            or _row_str(row, "continuation_authority_digest")
            != str(authority_digest(authority))
            or _row_str(row, "continuation_payload_lane_id")
            != str(_CONTINUATION_REFERENCE_LANE)
            or _row_int(row, "continuation_payload_sequence") != 0
            or _row_str(row, "continuation_payload_kind")
            != ConversationPayloadKind.CONTINUATION_REFERENCE.value
        ):
            raise ConversationStorageError()
        reference = self._durable_codec.decode_continuation_reference(
            await self._decrypt_payload_row(row)
        )
        if (
            str(reference.continuation_id) != _row_str(row, "continuation_id")
            or reference.state_revision
            != _row_int(row, "continuation_state_revision")
            or str(reference.digest) != _row_str(row, "continuation_digest")
            or continuation_definition_digest(reference.definition)
            != _row_str(row, "definition_digest")
            or continuation_revision_binding_digest(reference.revision_binding)
            != _row_str(row, "revision_binding_digest")
        ):
            raise ConversationStorageError()
        return reference

    async def retrieve(
        self,
        public_response_id: PublicResponseId,
        authority: AuthorityScope,
    ) -> ConversationResult:
        validate_identifier(public_response_id, "public_response_id")
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        await self._reach_store(StoreAwaitBoundary.RETRIEVE)
        row = await self._read_one(
            "public_response_load",
            _SELECT_PUBLIC_RESPONSE_SQL,
            (public_response_id, authority_digest(authority)),
        )
        if row is None or _row_bool(row, "tombstoned"):
            raise ConversationAuthorizationError()
        checkpoint_id = CheckpointId(_row_str(row, "checkpoint_id"))
        checkpoint = await self._load_checkpoint(checkpoint_id, authority)
        outputs = await self.retrieve_output_candidates(
            checkpoint_id, authority
        )
        mode = (
            ConversationMode.STORED
            if any(value.mode is ConversationMode.STORED for value in outputs)
            else ConversationMode.STATELESS
        )
        handle: ConversationHandle
        if mode is ConversationMode.STORED:
            handle = StoredConversationHandle(
                conversation_id=checkpoint.identity.conversation_id,
                checkpoint_id=checkpoint_id,
                branch_id=checkpoint.identity.branch_id,
                public_response_id=public_response_id,
            )
        else:
            handle = StatelessConversationHandle(
                conversation_id=checkpoint.identity.conversation_id,
                checkpoint_id=checkpoint_id,
                branch_id=checkpoint.identity.branch_id,
            )
        if checkpoint.integrity is None:
            raise ConversationStorageError()
        return ConversationResult(
            handle=handle,
            reasoning=outputs[-1].reasoning,
            checkpoint_digest=checkpoint.integrity.digest,
            lane_outputs=tuple(value.public_output for value in outputs),
            public_response_id=public_response_id,
        )

    async def fence_idempotency(
        self,
        identity: RequestIdempotencyIdentity,
        owner_token: str,
        *,
        ambiguous: bool,
    ) -> None:
        if (
            type(identity) is not RequestIdempotencyIdentity
            or type(ambiguous) is not bool
        ):
            raise ConversationValidationError()
        validate_identifier(owner_token, "owner_token")
        await self._reach_store(StoreAwaitBoundary.IDEMPOTENCY)
        await self._settle_owned_idempotency(
            identity,
            owner_token,
            ambiguous=ambiguous,
            remove=False,
            cleanup=False,
        )

    async def abandon_idempotency(
        self,
        identity: RequestIdempotencyIdentity,
        owner_token: str,
        *,
        ambiguous: bool,
    ) -> IdempotencySettlementResolution:
        if (
            type(identity) is not RequestIdempotencyIdentity
            or type(ambiguous) is not bool
        ):
            raise ConversationValidationError()
        validate_identifier(owner_token, "owner_token")
        await self._reach_store(StoreAwaitBoundary.ROLLBACK_BEGIN)
        return await self._settle_owned_idempotency(
            identity,
            owner_token,
            ambiguous=ambiguous,
            remove=not ambiguous,
            cleanup=True,
        )

    async def reconcile_idempotency(
        self,
        identity: RequestIdempotencyIdentity,
        owner_token: str,
        *,
        ambiguous: bool,
    ) -> IdempotencySettlementResolution:
        if (
            type(identity) is not RequestIdempotencyIdentity
            or type(ambiguous) is not bool
        ):
            raise ConversationValidationError()
        validate_identifier(owner_token, "owner_token")
        await self._reach_store(StoreAwaitBoundary.IDEMPOTENCY_RECONCILE_BEGIN)
        authority_key = str(authority_digest(identity.authority))
        now = await self._clock.now()
        _validate_time(now)

        async def operation(
            cursor: PgsqlCursor,
        ) -> IdempotencySettlementResolution:
            row = await self._fetchone(
                cursor,
                "idempotency_reconcile_select",
                _SELECT_IDEMPOTENCY_FOR_UPDATE_SQL,
                (authority_key, identity.operation.value, identity.key),
            )
            if row is None:
                return IdempotencySettlementResolution(
                    disposition=IdempotencySettlementDisposition.SETTLED
                )
            if _row_str(row, "owner_token") != owner_token or _row_str(
                row, "request_digest"
            ) != str(identity.request_digest):
                raise ConversationConflictError()
            await self._cleanup_owner(cursor, owner_token)
            if _row_str(row, "record_state") == "in_progress":
                if ambiguous:
                    await self._execute(
                        cursor,
                        "idempotency_reconcile_fence",
                        _UPDATE_IDEMPOTENCY_STATE_SQL,
                        (
                            IdempotencyRecordState.AMBIGUOUS.value,
                            now,
                            authority_key,
                            identity.operation.value,
                            identity.key,
                        ),
                    )
                else:
                    await self._execute(
                        cursor,
                        "idempotency_reconcile_delete",
                        _DELETE_IDEMPOTENCY_SQL,
                        (
                            authority_key,
                            identity.operation.value,
                            identity.key,
                        ),
                    )
            return IdempotencySettlementResolution(
                disposition=IdempotencySettlementDisposition.SETTLED
            )

        return await self._transaction("idempotency_reconcile", operation)

    async def reconcile_ambiguous_dispatch(
        self,
        request: AmbiguousDispatchReconciliationRequest,
    ) -> AmbiguousDispatchReconciliationResult:
        """Apply one explicit durable ambiguity decision."""
        if type(request) is not AmbiguousDispatchReconciliationRequest:
            raise ConversationValidationError()
        dispositions = AmbiguousDispatchReconciliationDisposition
        authority_key = str(authority_digest(request.authority))
        now = await self._clock.now()
        _validate_time(now)

        async def operation(
            cursor: PgsqlCursor,
        ) -> AmbiguousDispatchReconciliationResult:
            row = await self._fetchone(
                cursor,
                "ambiguous_dispatch_reconcile_select",
                _SELECT_IDEMPOTENCY_FOR_UPDATE_SQL,
                (
                    authority_key,
                    request.operation.value,
                    request.idempotency_key,
                ),
            )
            if row is None:
                disposition = dispositions.NOT_FOUND_OR_UNAUTHORIZED
            else:
                state = IdempotencyRecordState(_row_str(row, "record_state"))
                if state is IdempotencyRecordState.FAILED_NO_DISPATCH:
                    disposition = dispositions.ALREADY_RESOLVED_NO_DISPATCH
                elif state is not IdempotencyRecordState.AMBIGUOUS:
                    raise ConversationConflictError()
                elif (
                    request.resolution
                    is AmbiguousDispatchResolution.RETAIN_FENCE
                ):
                    disposition = dispositions.FENCE_RETAINED
                else:
                    await self._execute(
                        cursor,
                        "ambiguous_dispatch_reconcile_update",
                        _UPDATE_IDEMPOTENCY_STATE_SQL,
                        (
                            IdempotencyRecordState.FAILED_NO_DISPATCH.value,
                            now,
                            authority_key,
                            request.operation.value,
                            request.idempotency_key,
                        ),
                    )
                    disposition = dispositions.RESOLVED_NO_DISPATCH
            return AmbiguousDispatchReconciliationResult(
                disposition=disposition
            )

        return await self._transaction(
            "ambiguous_dispatch_reconcile",
            operation,
        )

    async def _settle_owned_idempotency(
        self,
        identity: RequestIdempotencyIdentity,
        owner_token: str,
        *,
        ambiguous: bool,
        remove: bool,
        cleanup: bool,
    ) -> IdempotencySettlementResolution:
        authority_key = str(authority_digest(identity.authority))
        now = await self._clock.now()
        _validate_time(now)

        async def operation(
            cursor: PgsqlCursor,
        ) -> IdempotencySettlementResolution:
            row = await self._fetchone(
                cursor,
                "idempotency_settle_select",
                _SELECT_IDEMPOTENCY_FOR_UPDATE_SQL,
                (authority_key, identity.operation.value, identity.key),
            )
            if (
                row is None
                or _row_str(row, "owner_token") != owner_token
                or _row_str(row, "request_digest")
                != str(identity.request_digest)
                or _row_str(row, "record_state") != "in_progress"
            ):
                raise ConversationConflictError()
            if cleanup:
                await self._cleanup_owner(cursor, owner_token)
            if remove:
                await self._execute(
                    cursor,
                    "idempotency_settle_delete",
                    _DELETE_IDEMPOTENCY_SQL,
                    (authority_key, identity.operation.value, identity.key),
                )
            else:
                record_state = (
                    IdempotencyRecordState.AMBIGUOUS.value
                    if ambiguous
                    else IdempotencyRecordState.FAILED_NO_DISPATCH.value
                )
                await self._execute(
                    cursor,
                    "idempotency_settle_update",
                    _UPDATE_IDEMPOTENCY_STATE_SQL,
                    (
                        record_state,
                        now,
                        authority_key,
                        identity.operation.value,
                        identity.key,
                    ),
                )
            return IdempotencySettlementResolution(
                disposition=IdempotencySettlementDisposition.SETTLED
            )

        return await self._transaction("idempotency_settle", operation)

    async def inspect_idempotency_settlement(
        self,
        identity: RequestIdempotencyIdentity,
        owner_token: str,
    ) -> IdempotencySettlementResolution:
        if type(identity) is not RequestIdempotencyIdentity:
            raise ConversationValidationError()
        validate_identifier(owner_token, "owner_token")
        await self._reach_store(StoreAwaitBoundary.IDEMPOTENCY_SETTLEMENT)
        row = await self._read_one(
            "idempotency_settlement_inspect",
            _SELECT_IDEMPOTENCY_SQL,
            (
                authority_digest(identity.authority),
                identity.operation.value,
                identity.key,
            ),
        )
        provisional = await self._read_one(
            "idempotency_settlement_provisional",
            _SELECT_PROVISIONAL_BY_OWNER_SQL,
            (owner_token,),
        )
        if row is None:
            return IdempotencySettlementResolution(
                disposition=(
                    IdempotencySettlementDisposition.OWNERSHIP_CONFLICT
                    if provisional is not None
                    else IdempotencySettlementDisposition.SETTLED
                )
            )
        if _row_str(row, "owner_token") != owner_token or _row_str(
            row, "request_digest"
        ) != str(identity.request_digest):
            return IdempotencySettlementResolution(
                disposition=IdempotencySettlementDisposition.OWNERSHIP_CONFLICT
            )
        if _row_str(row, "record_state") == "in_progress":
            return IdempotencySettlementResolution(
                disposition=IdempotencySettlementDisposition.LEASED,
                lease_expires_at=_row_datetime(row, "lease_expires_at"),
                lease_owner_token=owner_token,
            )
        return IdempotencySettlementResolution(
            disposition=(
                IdempotencySettlementDisposition.OWNERSHIP_CONFLICT
                if provisional is not None
                else IdempotencySettlementDisposition.SETTLED
            )
        )

    async def rollback_attempt(self, owner_token: str) -> None:
        validate_identifier(owner_token, "owner_token")
        await self._reach_store(StoreAwaitBoundary.ROLLBACK_BEGIN)

        async def operation(cursor: PgsqlCursor) -> None:
            await self._cleanup_owner(cursor, owner_token)

        await self._transaction("attempt_rollback", operation)

    async def _cleanup_owner(
        self,
        cursor: PgsqlCursor,
        owner_token: str,
    ) -> None:
        await self._execute(
            cursor,
            "attempt_delete_provisional",
            _DELETE_PROVISIONAL_OWNER_SQL,
            (owner_token,),
        )
        await self._execute(
            cursor,
            "attempt_delete_staging",
            _DELETE_EXECUTION_STAGING_ALL_OWNER_SQL,
            (owner_token,),
        )

    async def prepare_deletion(
        self,
        public_response_id: PublicResponseId,
        authority: AuthorityScope,
    ) -> LocalDeletionPreparation:
        """Resolve one authorized deletion without disclosing private state."""
        validate_identifier(public_response_id, "public_response_id")
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        await self._reach_store(StoreAwaitBoundary.PREPARE_DELETE)
        authority_key = authority_digest(authority)
        public = await self._read_one(
            "deletion_prepare_public",
            _SELECT_PUBLIC_RESPONSE_SQL,
            (public_response_id, authority_key),
        )
        if public is not None:
            tombstoned = _row_bool(public, "tombstoned")
            checkpoint = await self._load_checkpoint_lifecycle(
                CheckpointId(_row_str(public, "checkpoint_id")),
                authority,
                (
                    CheckpointLifecycle.TOMBSTONED
                    if tombstoned
                    else CheckpointLifecycle.COMMITTED
                ),
            )
            return LocalDeletionPreparation(
                state=(
                    LocalDeletionState.TOMBSTONED
                    if tombstoned
                    else LocalDeletionState.ACTIVE
                ),
                checkpoint=checkpoint,
            )
        terminal = await self._read_one(
            "deletion_prepare_terminal",
            _SELECT_DELETION_TERMINAL_SQL,
            (public_response_id, authority_key),
        )
        if (
            terminal is None
            or _row_str(terminal, "terminal_state")
            != CheckpointLifecycle.DELETED.value
        ):
            raise ConversationAuthorizationError()
        return LocalDeletionPreparation(
            state=LocalDeletionState.DELETED,
            checkpoint=None,
        )

    async def tombstone(
        self,
        public_response_id: PublicResponseId,
        authority: AuthorityScope,
        at: datetime,
    ) -> ConversationCheckpoint:
        validate_identifier(public_response_id, "public_response_id")
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        _validate_time(at)
        await self._reach_store(StoreAwaitBoundary.TOMBSTONE)
        public = await self._read_one(
            "tombstone_public_load",
            _SELECT_PUBLIC_RESPONSE_SQL,
            (public_response_id, authority_digest(authority)),
        )
        if public is None:
            raise ConversationAuthorizationError()
        checkpoint_id = CheckpointId(_row_str(public, "checkpoint_id"))
        if _row_bool(public, "tombstoned"):
            raise ConversationAuthorizationError()
        try:
            checkpoint = await self._load_checkpoint(checkpoint_id, authority)
        except ConversationAuthorizationError:
            return await self._load_checkpoint_lifecycle(
                checkpoint_id,
                authority,
                CheckpointLifecycle.TOMBSTONED,
            )
        tombstone = with_checkpoint_integrity(
            replace(
                checkpoint,
                lifecycle=CheckpointLifecycle.TOMBSTONED,
                timestamps=replace(
                    checkpoint.timestamps,
                    tombstoned_at=at,
                ),
            )
        )
        envelope, key = await self._prepare_lifecycle_envelope(tombstone)
        authority_key = str(authority_digest(authority))

        async def operation(cursor: PgsqlCursor) -> bool:
            row = await self._fetchone(
                cursor,
                "tombstone_checkpoint_lock",
                _SELECT_CHECKPOINT_FOR_UPDATE_SQL,
                (checkpoint_id,),
            )
            response = await self._fetchone(
                cursor,
                "tombstone_public_lock",
                _SELECT_PUBLIC_RESPONSE_FOR_UPDATE_SQL,
                (public_response_id,),
            )
            if (
                row is None
                or response is None
                or _row_str(row, "authority_digest") != authority_key
                or _row_str(response, "authority_digest") != authority_key
                or _row_str(response, "checkpoint_id") != str(checkpoint_id)
            ):
                raise ConversationConflictError()
            response_tombstoned = _row_bool(response, "tombstoned")
            lifecycle = _row_str(row, "lifecycle_state")
            if response_tombstoned and lifecycle == "tombstoned":
                return False
            if response_tombstoned or lifecycle != "committed":
                raise ConversationConflictError()
            await self._synchronize_write_key(cursor, authority_key, key)
            await self._replace_checkpoint_envelope(cursor, envelope)
            await self._execute(
                cursor,
                "tombstone_checkpoint_update",
                _TOMBSTONE_CHECKPOINT_SQL,
                (at, checkpoint_id, authority_key),
            )
            await self._execute(
                cursor,
                "tombstone_public_update",
                _TOMBSTONE_PUBLIC_RESPONSE_SQL,
                (at, public_response_id, authority_key),
            )
            await self._execute(
                cursor,
                "tombstone_publication_outbox_delete",
                _DELETE_OUTBOX_CHECKPOINT_SQL,
                (checkpoint_id,),
            )
            await self._execute(
                cursor,
                "tombstone_heads",
                _TOMBSTONE_HEADS_SQL,
                (at, checkpoint_id),
            )
            await self._insert_deletion_reconciliation(
                cursor,
                checkpoint_id,
                authority_key,
            )
            await self._upsert_terminal(
                cursor,
                checkpoint_id,
                public_response_id,
                CheckpointLifecycle.TOMBSTONED,
                at,
            )
            return True

        changed = await self._transaction("checkpoint_tombstone", operation)
        if not changed:
            return await self._load_checkpoint_lifecycle(
                checkpoint_id,
                authority,
                CheckpointLifecycle.TOMBSTONED,
            )
        return tombstone

    async def _prepare_lifecycle_envelope(
        self,
        checkpoint: ConversationCheckpoint,
    ) -> tuple[_PreparedPayload, ConversationDataKey]:
        encoded = self._checkpoint_codec.encode(checkpoint)
        self._validate_checkpoint_limits(checkpoint, len(encoded))
        digest = authority_digest(checkpoint.authority)
        key = await self._key_resolver.current_write_key(digest)
        if key.status is not ConversationKeyStatus.CURRENT:
            raise ConversationKeyPolicyError()
        envelope = await self._encrypt_payload(
            encoded,
            key=key,
            authority=digest,
            conversation_id=str(checkpoint.identity.conversation_id),
            checkpoint_id=checkpoint.identity.checkpoint_id,
            lane_id=_CHECKPOINT_ENVELOPE_LANE,
            sequence=0,
            kind=ConversationPayloadKind.CHECKPOINT,
            codec_version=CHECKPOINT_CODEC_VERSION,
        )
        return envelope, key

    async def _replace_checkpoint_envelope(
        self,
        cursor: PgsqlCursor,
        envelope: _PreparedPayload,
    ) -> None:
        await self._execute(
            cursor,
            "checkpoint_envelope_reference_delete",
            _DELETE_ENVELOPE_REFERENCE_SQL,
            (envelope.associated_data.checkpoint_id,),
        )
        await self._insert_payload(cursor, envelope)

    async def _insert_deletion_reconciliation(
        self,
        cursor: PgsqlCursor,
        checkpoint_id: CheckpointId,
        authority_key: str,
    ) -> None:
        lanes = await self._fetchall(
            cursor,
            "deletion_reconciliation_lanes",
            _SELECT_STORED_LANES_SQL,
            (checkpoint_id,),
        )
        for lane in lanes:
            await self._execute(
                cursor,
                "deletion_reconciliation_insert",
                _INSERT_RECONCILIATION_SQL,
                (
                    f"reconciliation-{uuid4().hex}",
                    checkpoint_id,
                    _row_str(lane, "lane_id"),
                    authority_key,
                    _row_str(lane, "conversation_id"),
                    0,
                    ConversationPayloadKind.DELETION_TARGET.value,
                    _row_str(lane, "payload_id"),
                    "delete_upstream",
                ),
            )

    async def delete(
        self,
        public_response_id: PublicResponseId,
        authority: AuthorityScope,
        at: datetime,
    ) -> None:
        validate_identifier(public_response_id, "public_response_id")
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        _validate_time(at)
        await self._reach_store(StoreAwaitBoundary.DELETE)
        authority_key = str(authority_digest(authority))

        async def operation(cursor: PgsqlCursor) -> None:
            response = await self._fetchone(
                cursor,
                "delete_public_lock",
                _SELECT_PUBLIC_RESPONSE_FOR_UPDATE_SQL,
                (public_response_id,),
            )
            if (
                response is None
                or _row_str(response, "authority_digest") != authority_key
            ):
                raise ConversationAuthorizationError()
            if not _row_bool(response, "tombstoned"):
                raise ConversationTransitionError()
            checkpoint_id = CheckpointId(_row_str(response, "checkpoint_id"))
            checkpoint = await self._fetchone(
                cursor,
                "delete_checkpoint_lock",
                _SELECT_CHECKPOINT_FOR_UPDATE_SQL,
                (checkpoint_id,),
            )
            if (
                checkpoint is None
                or _row_str(checkpoint, "lifecycle_state") != "tombstoned"
            ):
                raise ConversationAuthorizationError()
            lanes = await self._fetchall(
                cursor,
                "delete_reconciliation_lock",
                _SELECT_STORED_LANES_FOR_DELETE_SQL,
                (checkpoint_id,),
            )
            if any(
                _row_str(lane, "upstream_deletion_state")
                not in {"succeeded", "unsupported"}
                for lane in lanes
            ):
                raise ConversationTransitionError()
            await self._execute(
                cursor,
                "delete_checkpoint_payload_refs",
                _DELETE_ALL_PAYLOAD_REFERENCES_SQL,
                (checkpoint_id,),
            )
            await self._execute(
                cursor,
                "delete_checkpoint_idempotency",
                _DELETE_IDEMPOTENCY_CHECKPOINT_SQL,
                (checkpoint_id,),
            )
            await self._execute(
                cursor,
                "delete_public_response",
                _DELETE_PUBLIC_RESPONSE_SQL,
                (public_response_id, authority_key),
            )
            await self._execute(
                cursor,
                "delete_checkpoint_update",
                _DELETE_CHECKPOINT_LOGICALLY_SQL,
                (at, checkpoint_id, authority_key),
            )
            await self._upsert_terminal(
                cursor,
                checkpoint_id,
                public_response_id,
                CheckpointLifecycle.DELETED,
                at,
            )

        await self._transaction("checkpoint_delete", operation)

    async def list_checkpoints(
        self,
        authority: AuthorityScope,
        *,
        cursor: CheckpointId | None,
        limit: int,
    ) -> CheckpointPage:
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        if cursor is not None:
            validate_identifier(cursor, "cursor")
        if (
            type(limit) is not int
            or limit <= 0
            or limit > self._policy.limits.max_page_size
        ):
            raise ConversationLimitError()
        await self._reach_store(StoreAwaitBoundary.LIST)
        rows = await self._read_all(
            "checkpoint_list",
            _LIST_CHECKPOINTS_SQL,
            (
                authority_digest(authority),
                str(cursor) if cursor is not None else "",
                limit + 1,
            ),
        )
        selected = rows[:limit]
        checkpoints: list[ConversationCheckpoint] = []
        for row in selected:
            checkpoints.append(
                await self._load_checkpoint(
                    CheckpointId(_row_str(row, "checkpoint_id")), authority
                )
            )
        return CheckpointPage(
            checkpoints=tuple(checkpoints),
            next_cursor=(
                CheckpointId(_row_str(selected[-1], "checkpoint_id"))
                if len(rows) > limit and selected
                else None
            ),
        )

    async def sweep(self, now: datetime, *, limit: int) -> SweepReceipt:
        _validate_time(now)
        if (
            type(limit) is not int
            or limit <= 0
            or limit > (self._policy.max_batch_size)
        ):
            raise ConversationLimitError()
        await self._reach_store(StoreAwaitBoundary.SWEEP)

        async def operation(cursor: PgsqlCursor) -> SweepReceipt:
            expired_rows = await self._fetchall(
                cursor,
                "sweep_expired_existing",
                _SELECT_EXPIRED_FOR_DELETE_SQL,
                (limit,),
            )
            deleted = 0
            for row in expired_rows:
                checkpoint_id = CheckpointId(_row_str(row, "checkpoint_id"))
                await self._expire_delete(cursor, checkpoint_id, now)
                deleted += 1
            remaining = limit - deleted
            newly_expired: Sequence[PgsqlRow] = ()
            if remaining > 0:
                newly_expired = await self._fetchall(
                    cursor,
                    "sweep_select_committed",
                    _SELECT_COMMITTED_EXPIRY_SQL,
                    (now, remaining),
                )
            for row in newly_expired:
                checkpoint_id = CheckpointId(_row_str(row, "checkpoint_id"))
                await self._execute(
                    cursor,
                    "sweep_mark_expired",
                    _MARK_CHECKPOINT_EXPIRED_SQL,
                    (checkpoint_id,),
                )
                await self._insert_deletion_reconciliation(
                    cursor,
                    checkpoint_id,
                    _row_str(row, "authority_digest"),
                )
                await self._execute(
                    cursor,
                    "sweep_tombstone_public",
                    _EXPIRE_PUBLIC_RESPONSE_SQL,
                    (now, checkpoint_id),
                )
                await self._execute(
                    cursor,
                    "sweep_delete_outbox",
                    _DELETE_OUTBOX_CHECKPOINT_SQL,
                    (checkpoint_id,),
                )
                await self._upsert_terminal(
                    cursor,
                    checkpoint_id,
                    None,
                    CheckpointLifecycle.EXPIRED,
                    now,
                )
            return SweepReceipt(
                expired=len(newly_expired),
                deleted=deleted,
            )

        return await self._transaction("retention_sweep", operation)

    async def _expire_delete(
        self,
        cursor: PgsqlCursor,
        checkpoint_id: CheckpointId,
        at: datetime,
    ) -> None:
        await self._execute(
            cursor,
            "sweep_delete_refs",
            _DELETE_ALL_PAYLOAD_REFERENCES_SQL,
            (checkpoint_id,),
        )
        await self._execute(
            cursor,
            "sweep_delete_public",
            _DELETE_PUBLIC_RESPONSE_CHECKPOINT_SQL,
            (checkpoint_id,),
        )
        await self._execute(
            cursor,
            "sweep_delete_idempotency",
            _DELETE_IDEMPOTENCY_CHECKPOINT_SQL,
            (checkpoint_id,),
        )
        await self._execute(
            cursor,
            "sweep_logical_delete",
            _DELETE_EXPIRED_CHECKPOINT_SQL,
            (at, checkpoint_id),
        )
        await self._upsert_terminal(
            cursor,
            checkpoint_id,
            None,
            CheckpointLifecycle.DELETED,
            at,
        )

    async def prune(self, now: datetime, *, limit: int) -> PruneReceipt:
        _validate_time(now)
        if (
            type(limit) is not int
            or limit <= 0
            or limit > (self._policy.max_batch_size)
        ):
            raise ConversationLimitError()
        await self._reach_store(StoreAwaitBoundary.PRUNE)

        async def operation(cursor: PgsqlCursor) -> PruneReceipt:
            outbox_rows = await self._fetchall(
                cursor,
                "prune_outbox_select",
                _SELECT_PUBLISHED_OUTBOX_PRUNE_SQL,
                (now, limit),
            )
            for row in outbox_rows:
                await self._execute(
                    cursor,
                    "prune_outbox_delete",
                    _DELETE_OUTBOX_SQL,
                    (_row_str(row, "intent_id"),),
                )
            remaining = limit - len(outbox_rows)
            idempotency_rows: Sequence[PgsqlRow] = ()
            if remaining > 0:
                idempotency_rows = await self._fetchall(
                    cursor,
                    "prune_idempotency_select",
                    _SELECT_IDEMPOTENCY_PRUNE_SQL,
                    (remaining,),
                )
            for row in idempotency_rows:
                await self._execute(
                    cursor,
                    "prune_idempotency_delete",
                    _DELETE_IDEMPOTENCY_SQL,
                    (
                        _row_str(row, "authority_digest"),
                        _row_str(row, "operation"),
                        _row_str(row, "idempotency_key"),
                    ),
                )
            return PruneReceipt(
                outbox_records=len(outbox_rows),
                idempotency_records=len(idempotency_rows),
            )

        return await self._transaction("operational_prune", operation)

    async def _upsert_terminal(
        self,
        cursor: PgsqlCursor,
        checkpoint_id: CheckpointId,
        public_response_id: PublicResponseId | None,
        lifecycle: CheckpointLifecycle,
        at: datetime,
    ) -> None:
        await self._lock_global_capacity(cursor)
        count_row = await self._fetchone(
            cursor,
            "terminal_count",
            _COUNT_TERMINAL_SQL,
            None,
        )
        if count_row is None:
            raise ConversationStorageError()
        if _row_int(count_row, "record_count") >= (
            self._policy.limits.max_terminal_metadata
        ):
            await self._execute(
                cursor,
                "terminal_evict_oldest",
                _DELETE_OLDEST_TERMINAL_SQL,
                None,
            )
        await self._execute(
            cursor,
            "terminal_upsert",
            _UPSERT_TERMINAL_SQL,
            (checkpoint_id, public_response_id, lifecycle.value, at),
        )

    async def claim_outbox(
        self,
        target: OutboxClaimTarget,
    ) -> OutboxClaimResolution:
        if type(target) is not OutboxClaimTarget:
            raise ConversationValidationError()
        now = await self._clock.now()
        _validate_time(now)
        await self._reach_store(StoreAwaitBoundary.OUTBOX_CLAIM)
        preflight = await self._read_one(
            "outbox_claim_preflight",
            _SELECT_OUTBOX_SQL,
            (target.intent_id,),
        )
        if not _outbox_target_matches(preflight, target):
            return OutboxClaimResolution(
                disposition=OutboxClaimDisposition.NOT_FOUND_OR_UNAUTHORIZED
            )
        outputs = await self.retrieve_output_candidates(
            target.checkpoint_id,
            target.authority,
        )

        async def operation(cursor: PgsqlCursor) -> OutboxClaimResolution:
            row = await self._fetchone(
                cursor,
                "outbox_claim_select",
                _SELECT_OUTBOX_FOR_UPDATE_SQL,
                (target.intent_id,),
            )
            if not _outbox_target_matches(row, target):
                return OutboxClaimResolution(
                    disposition=(
                        OutboxClaimDisposition.NOT_FOUND_OR_UNAUTHORIZED
                    )
                )
            assert row is not None
            state = OutboxState(_row_str(row, "outbox_state"))
            if state is OutboxState.PUBLISHED:
                return OutboxClaimResolution(
                    disposition=OutboxClaimDisposition.ALREADY_PUBLISHED
                )
            if state is OutboxState.CLAIMED:
                lease = _row_optional_datetime(row, "lease_expires_at")
                if lease is not None and lease > now:
                    return OutboxClaimResolution(
                        disposition=OutboxClaimDisposition.ACTIVELY_LEASED
                    )
            owner_token = f"outbox-owner-{uuid4().hex}"
            lease_expires_at = now + timedelta(
                seconds=self._policy.limits.outbox_lease_seconds
            )
            await self._execute(
                cursor,
                "outbox_claim_update",
                _CLAIM_OUTBOX_SQL,
                (
                    owner_token,
                    lease_expires_at,
                    target.intent_id,
                ),
            )
            record = _outbox_row_to_record(
                row,
                outputs,
                state=OutboxState.CLAIMED,
                attempts=_row_int(row, "attempts") + 1,
                lease_owner=owner_token,
                lease_expires_at=lease_expires_at,
                published_at=None,
            )
            return OutboxClaimResolution(
                disposition=OutboxClaimDisposition.CLAIMED,
                record=record,
            )

        return await self._transaction("outbox_claim", operation)

    def create_outbox_recovery_worker(
        self,
        authority: AuthorityScope,
    ) -> ConversationOutboxRecoveryWorker:
        """Create one trusted authority-isolated recovery worker."""
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        store = self

        @final
        class AuthorityOutboxRecoveryWorker:
            async def claim(self, *, limit: int) -> OutboxRecoveryBatch:
                return await store._claim_pending_outbox(
                    authority,
                    limit=limit,
                )

            async def acknowledge(self, record: OutboxRecord) -> None:
                target, owner_token = self._settlement(record)
                await store.acknowledge_outbox(target, owner_token)

            async def release(self, record: OutboxRecord) -> None:
                target, owner_token = self._settlement(record)
                await store.release_outbox(target, owner_token)

            @staticmethod
            def _settlement(
                record: OutboxRecord,
            ) -> tuple[OutboxClaimTarget, str]:
                if (
                    type(record) is not OutboxRecord
                    or record.state is not OutboxState.CLAIMED
                    or record.lease_owner is None
                ):
                    raise ConversationValidationError()
                return (
                    OutboxClaimTarget(
                        authority=authority,
                        checkpoint_id=record.intent.checkpoint_id,
                        public_response_id=record.intent.public_response_id,
                        intent_id=record.intent.intent_id,
                    ),
                    record.lease_owner,
                )

        return AuthorityOutboxRecoveryWorker()

    async def _claim_pending_outbox(
        self,
        authority: AuthorityScope,
        *,
        limit: int,
    ) -> OutboxRecoveryBatch:
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        if (
            type(limit) is not int
            or limit <= 0
            or limit > self._policy.max_batch_size
        ):
            raise ConversationLimitError()
        now = await self._clock.now()
        _validate_time(now)
        await self._reach_store(StoreAwaitBoundary.OUTBOX_RECOVERY_CLAIM)
        authority_key = str(authority_digest(authority))

        async def operation(
            cursor: PgsqlCursor,
        ) -> tuple[tuple[PgsqlRow, str, datetime], ...]:
            rows = await self._fetchall(
                cursor,
                "outbox_recovery_select",
                _SELECT_RECOVERABLE_OUTBOX_SQL,
                (authority_key, now, limit),
            )
            claimed: list[tuple[PgsqlRow, str, datetime]] = []
            for row in rows:
                owner_token = f"outbox-owner-{uuid4().hex}"
                lease_expires_at = now + timedelta(
                    seconds=self._policy.limits.outbox_lease_seconds
                )
                await self._execute(
                    cursor,
                    "outbox_recovery_claim",
                    _CLAIM_OUTBOX_SQL,
                    (
                        owner_token,
                        lease_expires_at,
                        _row_str(row, "intent_id"),
                    ),
                )
                claimed.append((row, owner_token, lease_expires_at))
            return tuple(claimed)

        rows = await self._transaction("outbox_recovery_claim", operation)
        records: list[OutboxRecord] = []
        for row, owner_token, lease_expires_at in rows:
            checkpoint_id = CheckpointId(_row_str(row, "checkpoint_id"))
            outputs = await self.retrieve_output_candidates(
                checkpoint_id,
                authority,
            )
            records.append(
                _outbox_row_to_record(
                    row,
                    outputs,
                    state=OutboxState.CLAIMED,
                    attempts=_row_int(row, "attempts") + 1,
                    lease_owner=owner_token,
                    lease_expires_at=lease_expires_at,
                    published_at=None,
                )
            )
        return OutboxRecoveryBatch(
            disposition=(
                OutboxRecoveryDisposition.CLAIMED
                if records
                else OutboxRecoveryDisposition.EMPTY
            ),
            limit=limit,
            records=tuple(records),
        )

    async def acknowledge_outbox(
        self,
        target: OutboxClaimTarget,
        owner_token: str,
    ) -> None:
        if type(target) is not OutboxClaimTarget:
            raise ConversationValidationError()
        validate_identifier(owner_token, "owner_token")
        at = await self._clock.now()
        _validate_time(at)
        await self._reach_store(StoreAwaitBoundary.OUTBOX_ACKNOWLEDGE)

        async def operation(cursor: PgsqlCursor) -> None:
            row = await self._fetchone(
                cursor,
                "outbox_ack_select",
                _SELECT_OUTBOX_FOR_UPDATE_SQL,
                (target.intent_id,),
            )
            if not _outbox_target_matches(row, target):
                raise ConversationConflictError()
            assert row is not None
            state = OutboxState(_row_str(row, "outbox_state"))
            if state is OutboxState.PUBLISHED:
                return
            if (
                state is not OutboxState.CLAIMED
                or _row_optional_str(row, "lease_owner") != owner_token
            ):
                raise ConversationConflictError()
            await self._execute(
                cursor,
                "outbox_ack_update",
                _ACKNOWLEDGE_OUTBOX_SQL,
                (at, target.intent_id, owner_token),
            )

        await self._transaction("outbox_acknowledge", operation)

    async def release_outbox(
        self,
        target: OutboxClaimTarget,
        owner_token: str,
    ) -> None:
        if type(target) is not OutboxClaimTarget:
            raise ConversationValidationError()
        validate_identifier(owner_token, "owner_token")
        await self._reach_store(StoreAwaitBoundary.OUTBOX_RELEASE)

        async def operation(cursor: PgsqlCursor) -> None:
            row = await self._fetchone(
                cursor,
                "outbox_release_select",
                _SELECT_OUTBOX_FOR_UPDATE_SQL,
                (target.intent_id,),
            )
            if not _outbox_target_matches(row, target):
                raise ConversationConflictError()
            assert row is not None
            state = OutboxState(_row_str(row, "outbox_state"))
            if state is OutboxState.PUBLISHED:
                return
            if (
                state is not OutboxState.CLAIMED
                or _row_optional_str(row, "lease_owner") != owner_token
            ):
                raise ConversationConflictError()
            await self._execute(
                cursor,
                "outbox_release_update",
                _RELEASE_OUTBOX_SQL,
                (target.intent_id, owner_token),
            )

        await self._transaction("outbox_release", operation)

    async def claim_reconciliation(
        self,
        authority: AuthorityScope,
        *,
        limit: int,
    ) -> tuple[ReconciliationWorkRecord, ...]:
        """Claim bounded upstream-deletion or key-rewrap work."""
        return await self._claim_reconciliation(
            authority,
            limit=limit,
            provider_lifecycle_only=False,
        )

    async def _claim_reconciliation(
        self,
        authority: AuthorityScope,
        *,
        limit: int,
        provider_lifecycle_only: bool,
    ) -> tuple[ReconciliationWorkRecord, ...]:
        """Claim one bounded exact reconciliation work class."""
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        if (
            type(limit) is not int
            or limit <= 0
            or limit > self._policy.max_batch_size
            or type(provider_lifecycle_only) is not bool
        ):
            raise ConversationLimitError()
        now = await self._clock.now()
        _validate_time(now)
        authority_key = str(authority_digest(authority))

        async def operation(
            cursor: PgsqlCursor,
        ) -> tuple[ReconciliationWorkRecord, ...]:
            rows = await self._fetchall(
                cursor,
                (
                    "provider_lifecycle_claim_select"
                    if provider_lifecycle_only
                    else "reconciliation_claim_select"
                ),
                (
                    _SELECT_PROVIDER_LIFECYCLE_SQL
                    if provider_lifecycle_only
                    else _SELECT_RECONCILIATION_SQL
                ),
                (authority_key, now, limit),
            )
            result: list[ReconciliationWorkRecord] = []
            for row in rows:
                checkpoint_id = CheckpointId(_row_str(row, "checkpoint_id"))
                lane_id = ProviderLaneId(_row_str(row, "lane_id"))
                work_kind = _row_str(row, "work_kind")
                upstream_response_id = await self._reconciliation_target(row)
                owner_token = f"reconciliation-owner-{uuid4().hex}"
                lease_expires_at = now + timedelta(
                    seconds=self._policy.limits.outbox_lease_seconds
                )
                await self._execute(
                    cursor,
                    "reconciliation_claim_update",
                    _CLAIM_RECONCILIATION_SQL,
                    (
                        owner_token,
                        lease_expires_at,
                        _row_str(row, "reconciliation_id"),
                    ),
                )
                result.append(
                    ReconciliationWorkRecord(
                        reconciliation_id=_row_str(row, "reconciliation_id"),
                        checkpoint_id=checkpoint_id,
                        lane_id=lane_id,
                        work_kind=work_kind,
                        state=ReconciliationWorkState.CLAIMED,
                        attempts=_row_int(row, "attempts") + 1,
                        upstream_response_id=upstream_response_id,
                        lease_owner=owner_token,
                        lease_expires_at=lease_expires_at,
                        binding_digest=IntegrityDigest(
                            _row_str(row, "binding_digest")
                        ),
                        checkpoint_lifecycle=CheckpointLifecycle(
                            _row_str(row, "checkpoint_lifecycle_state")
                        ),
                    )
                )
            return tuple(result)

        return await self._transaction("reconciliation_claim", operation)

    async def _reconciliation_target(
        self,
        row: PgsqlRow,
    ) -> UpstreamResponseId | None:
        if _row_str(row, "work_kind") != "delete_upstream":
            return None
        checkpoint_id = CheckpointId(_row_str(row, "checkpoint_id"))
        lane_id = ProviderLaneId(_row_str(row, "lane_id"))
        authority_key = AuthorityDigest(_row_str(row, "authority_digest"))
        self._validate_payload_reference_row(
            row,
            checkpoint_id=checkpoint_id,
            authority=authority_key,
            kind=ConversationPayloadKind.DELETION_TARGET,
            lane_id=lane_id,
            sequence=0,
        )
        try:
            target = (await self._decrypt_payload_row(row)).decode("utf-8")
        except UnicodeError as error:
            raise ConversationStorageError() from error
        validate_identifier(target, "upstream_response_id")
        return UpstreamResponseId(target)

    async def claim_provider_lifecycle(
        self,
        authority: AuthorityScope,
        *,
        limit: int,
    ) -> tuple[ProviderLifecycleWorkRecord, ...]:
        """Claim bounded provider deletion work for one authority."""
        records = await self._claim_reconciliation(
            authority,
            limit=limit,
            provider_lifecycle_only=True,
        )
        result: list[ProviderLifecycleWorkRecord] = []
        for record in records:
            upstream_response_id = record.upstream_response_id
            binding_digest = record.binding_digest
            lifecycle = record.checkpoint_lifecycle
            if (
                record.work_kind != "delete_upstream"
                or upstream_response_id is None
                or binding_digest is None
                or lifecycle is None
            ):
                raise ConversationStorageError()
            origin = (
                ProviderLifecycleOrigin.COMMIT_QUARANTINE
                if str(record.checkpoint_id).startswith("quarantine-")
                else (
                    ProviderLifecycleOrigin.LOCAL_EXPIRY
                    if lifecycle is CheckpointLifecycle.EXPIRED
                    else ProviderLifecycleOrigin.LOCAL_TOMBSTONE
                )
            )
            result.append(
                ProviderLifecycleWorkRecord(
                    work_id=record.reconciliation_id,
                    checkpoint_id=record.checkpoint_id,
                    lane_id=record.lane_id,
                    binding_digest=binding_digest,
                    upstream_response_id=upstream_response_id,
                    origin=origin,
                    state=ProviderLifecycleWorkState.CLAIMED,
                    attempts=record.attempts,
                    lease_owner=record.lease_owner,
                    lease_expires_at=record.lease_expires_at,
                )
            )
        return tuple(result)

    async def quarantine_provider_checkpoint(
        self,
        request: ProviderQuarantineRequest,
    ) -> ProviderQuarantineReceipt:
        """Persist one private cleanup checkpoint and outbox atomically."""
        if type(request) is not ProviderQuarantineRequest:
            raise ConversationValidationError()
        candidates = (request.candidate, *request.additional_candidates)
        prepared = tuple(
            [
                await self._prepare_checkpoint(
                    candidate,
                    committed_at=request.created_at,
                    output_candidates=(),
                )
                for candidate in candidates
            ]
        )

        async def operation(cursor: PgsqlCursor) -> None:
            for item in prepared:
                checkpoint = item.checkpoint
                authority_key = str(authority_digest(checkpoint.authority))
                existing = await self._fetchone(
                    cursor,
                    "provider_quarantine_existing",
                    _SELECT_CHECKPOINT_FOR_UPDATE_SQL,
                    (checkpoint.identity.checkpoint_id,),
                )
                if existing is not None:
                    if _row_str(
                        existing, "authority_digest"
                    ) != authority_key or _row_str(
                        existing, "conversation_id"
                    ) != str(
                        checkpoint.identity.conversation_id
                    ):
                        raise ConversationConflictError()
                    payload = await self._fetchone(
                        cursor,
                        "provider_quarantine_existing_payload",
                        _SELECT_CHECKPOINT_PAYLOAD_SQL,
                        (checkpoint.identity.checkpoint_id, authority_key),
                    )
                    if (
                        payload is None
                        or _row_str(payload, "lifecycle_state")
                        != CheckpointLifecycle.COMMITTED.value
                    ):
                        raise ConversationStorageError()
                    self._validate_payload_reference_row(
                        payload,
                        checkpoint_id=checkpoint.identity.checkpoint_id,
                        authority=AuthorityDigest(authority_key),
                        kind=ConversationPayloadKind.CHECKPOINT,
                        lane_id=_CHECKPOINT_ENVELOPE_LANE,
                        sequence=0,
                    )
                    restored = self._checkpoint_codec.decode(
                        await self._decrypt_payload_row(payload)
                    )
                    if restored != checkpoint:
                        raise ConversationConflictError()
                    continue
                await self._insert_checkpoint(
                    cursor,
                    item,
                    enforce_capacity=False,
                )
                await self._insert_deletion_reconciliation(
                    cursor,
                    checkpoint.identity.checkpoint_id,
                    authority_key,
                )

        await self._transaction("provider_quarantine", operation)
        return ProviderQuarantineReceipt(
            checkpoint_id=prepared[0].checkpoint.identity.checkpoint_id,
            target_count=len(prepared),
        )

    async def acknowledge_provider_lifecycle(
        self,
        record: ProviderLifecycleWorkRecord,
        *,
        succeeded: bool,
    ) -> None:
        """Settle one exact provider deletion attempt."""
        if type(record) is not ProviderLifecycleWorkRecord:
            raise ConversationValidationError()
        await self.acknowledge_reconciliation(
            ReconciliationWorkRecord(
                reconciliation_id=record.work_id,
                checkpoint_id=record.checkpoint_id,
                lane_id=record.lane_id,
                work_kind="delete_upstream",
                state=ReconciliationWorkState.CLAIMED,
                attempts=record.attempts,
                upstream_response_id=record.upstream_response_id,
                lease_owner=record.lease_owner,
                lease_expires_at=record.lease_expires_at,
                binding_digest=record.binding_digest,
            ),
            succeeded=succeeded,
        )

    async def acknowledge_reconciliation(
        self,
        record: ReconciliationWorkRecord,
        *,
        succeeded: bool,
    ) -> None:
        """Settle one exact owner-leased reconciliation record."""
        if (
            type(record) is not ReconciliationWorkRecord
            or record.state is not ReconciliationWorkState.CLAIMED
            or record.lease_owner is None
            or type(succeeded) is not bool
        ):
            raise ConversationValidationError()
        at = await self._clock.now()
        _validate_time(at)

        async def operation(cursor: PgsqlCursor) -> None:
            row = await self._fetchone(
                cursor,
                "reconciliation_ack_select",
                _SELECT_RECONCILIATION_FOR_UPDATE_SQL,
                (record.reconciliation_id,),
            )
            if (
                row is None
                or _row_str(row, "work_state") != "claimed"
                or _row_optional_str(row, "lease_owner") != record.lease_owner
                or _row_str(row, "checkpoint_id") != str(record.checkpoint_id)
                or _row_str(row, "lane_id") != str(record.lane_id)
            ):
                raise ConversationConflictError()
            if (
                await self._reconciliation_target(row)
                != record.upstream_response_id
            ):
                raise ConversationConflictError()
            await self._execute(
                cursor,
                "reconciliation_ack_update",
                _ACKNOWLEDGE_RECONCILIATION_SQL,
                (
                    "completed" if succeeded else "failed",
                    at if succeeded else None,
                    record.reconciliation_id,
                    record.lease_owner,
                ),
            )
            if record.work_kind == "delete_upstream":
                await self._execute(
                    cursor,
                    "reconciliation_lane_update",
                    _UPDATE_LANE_RECONCILIATION_SQL,
                    (
                        "succeeded" if succeeded else "failed",
                        record.checkpoint_id,
                        record.lane_id,
                    ),
                )

        await self._transaction("reconciliation_acknowledge", operation)

    async def garbage_collect(
        self,
        *,
        limit: int,
    ) -> GarbageCollectionReceipt:
        """Delete a bounded set of authenticated unreferenced payloads."""
        if (
            type(limit) is not int
            or limit <= 0
            or limit > self._policy.max_batch_size
        ):
            raise ConversationLimitError()

        async def operation(cursor: PgsqlCursor) -> GarbageCollectionReceipt:
            rows = await self._fetchall(
                cursor,
                "garbage_collect_select",
                _SELECT_GARBAGE_PAYLOADS_SQL,
                (limit,),
            )
            for row in rows:
                await self._execute(
                    cursor,
                    "garbage_collect_delete",
                    _DELETE_GARBAGE_PAYLOAD_SQL,
                    (_row_str(row, "payload_id"),),
                )
            return GarbageCollectionReceipt(deleted_payloads=len(rows))

        return await self._transaction("payload_garbage_collect", operation)

    async def rotate_keys(
        self,
        authority: AuthorityScope,
        *,
        limit: int,
    ) -> KeyRotationReceipt:
        """Re-encrypt a bounded set of payloads under the current write key."""
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        if (
            type(limit) is not int
            or limit <= 0
            or limit > self._policy.max_batch_size
        ):
            raise ConversationLimitError()
        authority_key = authority_digest(authority)
        current = await self._key_resolver.current_write_key(authority_key)
        if current.status is not ConversationKeyStatus.CURRENT:
            raise ConversationKeyPolicyError()
        rows = await self._read_all(
            "key_rotation_candidates",
            _SELECT_KEY_ROTATION_PAYLOADS_SQL,
            (authority_key, current.key_id, current.revision, limit),
        )
        rotated: list[_RotatedPayload] = []
        for row in rows:
            plaintext = await self._decrypt_payload_row(row)
            associated_data = ConversationPayloadAssociatedData(
                authority_digest=authority_key,
                checkpoint_id=CheckpointId(_row_str(row, "checkpoint_id")),
                lane_id=ProviderLaneId(_row_str(row, "lane_id")),
                sequence=_row_int(row, "payload_sequence"),
                payload_kind=ConversationPayloadKind(
                    _row_str(row, "payload_kind")
                ),
                payload_schema_version=_row_int(row, "payload_schema_version"),
                codec_version=ConversationCodecVersion(
                    _row_int(row, "codec_version")
                ),
                key_id=current.key_id,
                key_revision=current.revision,
            )
            encrypted = await self._cipher.encrypt(
                plaintext,
                key=current,
                associated_data=associated_data,
            )
            rotated.append(
                _RotatedPayload(
                    prepared=_PreparedPayload(
                        payload_id=_row_str(row, "payload_id"),
                        conversation_id=_row_str(row, "conversation_id"),
                        associated_data=associated_data,
                        encrypted=encrypted,
                        plaintext_bytes=len(plaintext),
                    ),
                    previous_key_id=_row_str(row, "key_id"),
                    previous_key_revision=_row_int(row, "key_revision"),
                    previous_digest=_row_str(row, "authenticated_digest"),
                )
            )

        async def operation(cursor: PgsqlCursor) -> None:
            await self._synchronize_write_key(
                cursor, str(authority_key), current
            )
            for value in rotated:
                row = await self._fetchone(
                    cursor,
                    "key_rotation_payload_lock",
                    _SELECT_PAYLOAD_FOR_UPDATE_SQL,
                    (value.prepared.payload_id,),
                )
                if (
                    row is None
                    or _row_str(row, "authority_digest") != str(authority_key)
                    or _row_str(row, "key_id") != value.previous_key_id
                    or _row_int(row, "key_revision")
                    != value.previous_key_revision
                    or _row_str(row, "authenticated_digest")
                    != value.previous_digest
                ):
                    raise ConversationConflictError()
                encrypted = value.prepared.encrypted
                await self._execute(
                    cursor,
                    "key_rotation_payload_update",
                    _UPDATE_ROTATED_PAYLOAD_SQL,
                    (
                        encrypted.key_id,
                        encrypted.key_revision,
                        encrypted.algorithm,
                        encrypted.nonce,
                        encrypted.ciphertext,
                        encrypted.authenticated_digest,
                        value.prepared.payload_id,
                        value.previous_key_id,
                        value.previous_key_revision,
                        value.previous_digest,
                    ),
                )

        await self._transaction("key_rotation", operation)
        return KeyRotationReceipt(
            examined=len(rows),
            reencrypted=len(rotated),
        )

    async def retire_key(
        self,
        authority: AuthorityScope,
        *,
        key_id: str,
        revision: int,
        at: datetime,
    ) -> None:
        """Retire one grace key after every payload is re-encrypted."""
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        validate_identifier(key_id, "key_id")
        if type(revision) is not int or revision <= 0:
            raise ConversationValidationError()
        _validate_time(at)
        authority_key = authority_digest(authority)
        current = await self._key_resolver.current_write_key(authority_key)
        if current.key_id == key_id and current.revision == revision:
            raise ConversationKeyPolicyError()

        async def operation(cursor: PgsqlCursor) -> None:
            await self._synchronize_write_key(
                cursor, str(authority_key), current
            )
            row = await self._fetchone(
                cursor,
                "key_retire_select",
                _SELECT_KEY_REVISION_FOR_UPDATE_SQL,
                (authority_key, key_id, revision),
            )
            if row is None:
                raise ConversationKeyPolicyError()
            references = await self._fetchone(
                cursor,
                "key_retire_references",
                _COUNT_KEY_PAYLOAD_REFERENCES_SQL,
                (authority_key, key_id, revision),
            )
            if references is None or _row_int(references, "record_count") > 0:
                raise ConversationConflictError()
            await self._execute(
                cursor,
                "key_retire_update",
                _RETIRE_KEY_SQL,
                (at, authority_key, key_id, revision),
            )

        await self._transaction("key_retire", operation)

    async def close(self) -> StoreCloseResolution:
        if self._closed:
            return StoreCloseResolution(
                disposition=StoreCloseDisposition.CLOSED
            )
        await self._reach_store(StoreAwaitBoundary.CLOSE_BEGIN)
        if self._owns_database:
            close = getattr(self._database, "aclose", None)
            if close is not None:
                result = close()
                if isinstance(result, Awaitable):
                    await result
        self._opened = False
        self._closed = True
        if self._boundary_hook is not None:
            await self._boundary_hook.reach(StoreAwaitBoundary.CLOSE_SETTLED)
        return StoreCloseResolution(disposition=StoreCloseDisposition.CLOSED)

    async def inspect_close(self) -> StoreCloseResolution:
        if self._closed:
            if self._boundary_hook is not None:
                await self._boundary_hook.reach(
                    StoreAwaitBoundary.CLOSE_STATUS
                )
        else:
            await self._reach_store(StoreAwaitBoundary.CLOSE_STATUS)
        return StoreCloseResolution(
            disposition=(
                StoreCloseDisposition.CLOSED
                if self._closed
                else StoreCloseDisposition.OPEN
            )
        )

    async def _schema_readiness(self) -> PgsqlRow:
        revision = await self._read_one_unchecked(
            "schema_revision",
            _CHECK_SCHEMA_REVISION_SQL,
            None,
        )
        if (
            revision is None
            or _row_str(revision, "version_num")
            != CONVERSATION_PGSQL_HEAD_REVISION
        ):
            raise ConversationMigrationRequiredError()
        metadata = await self._read_one_unchecked(
            "schema_readiness",
            _CHECK_SCHEMA_READINESS_SQL,
            None,
        )
        if metadata is None:
            raise ConversationMigrationRequiredError()
        schema_version = _row_int(metadata, "schema_version")
        minimum_reader = _row_int(metadata, "minimum_reader_version")
        maximum_reader = _row_int(metadata, "maximum_reader_version")
        minimum_writer = _row_int(metadata, "minimum_writer_version")
        maximum_writer = _row_int(metadata, "maximum_writer_version")
        if (
            not self._policy.minimum_schema_version
            <= schema_version
            <= self._policy.maximum_schema_version
            or not minimum_reader
            <= self._policy.application_version
            <= maximum_reader
            or not minimum_writer
            <= self._policy.application_version
            <= maximum_writer
            or _row_int(metadata, "checkpoint_codec_version")
            != int(CHECKPOINT_CODEC_VERSION)
        ):
            raise ConversationMigrationRequiredError()
        return metadata

    async def _transaction(
        self,
        operation: str,
        callback: Callable[[PgsqlCursor], Awaitable[_T]],
    ) -> _T:
        self._ensure_open()
        validate_identifier(operation, "operation")
        await self._reach_fault(
            PgsqlConversationFaultBoundary.TRANSACTION_BEFORE,
            operation,
        )
        try:
            async with self._database.connection() as connection:
                async with connection.transaction():
                    async with connection.cursor() as cursor:
                        result = await callback(cursor)
                        await self._reach_fault(
                            PgsqlConversationFaultBoundary.COMMIT_BEFORE,
                            operation,
                        )
            await self._reach_fault(
                PgsqlConversationFaultBoundary.COMMIT_AFTER,
                operation,
            )
            return result
        except (KeyboardInterrupt, SystemExit, CancelledError):
            raise
        except ConversationError:
            raise
        except BaseException as error:
            failure = classify_pgsql_error(error, operation=operation)
            if failure.category is PgsqlFailureCategory.UNIQUE_CONFLICT:
                raise ConversationConflictError() from None
            raise ConversationStorageError() from None

    async def _read_one(
        self,
        operation: str,
        query: str,
        parameters: tuple[object, ...] | None,
    ) -> PgsqlRow | None:
        self._ensure_open()
        return await self._read_one_unchecked(operation, query, parameters)

    async def _read_one_unchecked(
        self,
        operation: str,
        query: str,
        parameters: tuple[object, ...] | None,
    ) -> PgsqlRow | None:
        try:
            async with self._database.connection() as connection:
                async with connection.cursor() as cursor:
                    return await self._fetchone(
                        cursor, operation, query, parameters
                    )
        except (KeyboardInterrupt, SystemExit, CancelledError):
            raise
        except ConversationError:
            raise
        except BaseException:
            raise ConversationStorageError() from None

    async def _read_all(
        self,
        operation: str,
        query: str,
        parameters: tuple[object, ...] | None,
    ) -> Sequence[PgsqlRow]:
        self._ensure_open()
        try:
            async with self._database.connection() as connection:
                async with connection.cursor() as cursor:
                    return await self._fetchall(
                        cursor, operation, query, parameters
                    )
        except (KeyboardInterrupt, SystemExit, CancelledError):
            raise
        except ConversationError:
            raise
        except BaseException:
            raise ConversationStorageError() from None

    async def _execute(
        self,
        cursor: PgsqlCursor,
        operation: str,
        query: str,
        parameters: tuple[object, ...] | None,
    ) -> None:
        await self._reach_fault(
            PgsqlConversationFaultBoundary.SQL_BEFORE,
            operation,
        )
        await cursor.execute(query, parameters)
        await self._reach_fault(
            PgsqlConversationFaultBoundary.SQL_AFTER,
            operation,
        )

    async def _fetchone(
        self,
        cursor: PgsqlCursor,
        operation: str,
        query: str,
        parameters: tuple[object, ...] | None,
    ) -> PgsqlRow | None:
        await self._execute(cursor, operation, query, parameters)
        return await cursor.fetchone()

    async def _fetchall(
        self,
        cursor: PgsqlCursor,
        operation: str,
        query: str,
        parameters: tuple[object, ...] | None,
    ) -> Sequence[PgsqlRow]:
        await self._execute(cursor, operation, query, parameters)
        return await cursor.fetchall()

    async def _reach_store(self, boundary: StoreAwaitBoundary) -> None:
        self._ensure_not_closed()
        if self._boundary_hook is not None:
            await self._boundary_hook.reach(boundary)

    async def _reach_fault(
        self,
        boundary: PgsqlConversationFaultBoundary,
        operation: str,
    ) -> None:
        await self._fault_hook.reach(
            PgsqlConversationFaultPoint(
                boundary=boundary,
                operation=operation,
                ordinal=next(self._fault_ordinals),
            )
        )

    def _ensure_open(self) -> None:
        self._ensure_not_closed()
        if not self._opened:
            raise ConversationFeatureUnavailableError()

    def _ensure_not_closed(self) -> None:
        if self._closed:
            raise ConversationStorageError()


def _reservation_matches_stage(
    row: PgsqlRow,
    stage: ProviderLaneExecutionStage,
) -> bool:
    identity = stage.identity
    return (
        _row_str(row, "checkpoint_id") == str(identity.checkpoint_id)
        and _row_str(row, "conversation_id") == str(identity.conversation_id)
        and _row_str(row, "logical_turn_id") == str(identity.logical_turn_id)
        and _row_str(row, "execution_segment_id")
        == str(identity.execution_segment_id)
        and _row_str(row, "branch_id") == str(identity.branch_id)
        and _row_int(row, "checkpoint_sequence") == identity.sequence
        and _row_optional_str(row, "parent_checkpoint_id")
        == (
            str(identity.parent_checkpoint_id)
            if identity.parent_checkpoint_id is not None
            else None
        )
        and _row_optional_int(row, "parent_sequence")
        == identity.parent_sequence
        and _row_str(row, "binding_digest")
        == str(stage.binding.integrity_digest)
        and _row_str(row, "lane_mode") == stage.mode.value
        and _row_str(row, "output_scope") == stage.scope.value
    )


def _validate_tool_recovery_checkpoint(
    admission: DurableToolRecoveryAdmission,
    execution: ConversationExecutionReservation,
    checkpoint: ConversationCheckpoint,
) -> None:
    """Validate one immutable checkpoint as the exact recovery suffix."""
    segments = checkpoint.content.execution_segments
    integrity = checkpoint.integrity
    if (
        checkpoint.identity.checkpoint_id != admission.checkpoint_id
        or checkpoint.kind is not CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
        or checkpoint.lifecycle is not CheckpointLifecycle.COMMITTED
        or checkpoint.authority != admission.idempotency.authority
        or integrity is None
        or integrity.digest != admission.checkpoint_integrity
        or len(segments) != admission.segment_count
        or not segments
        or segments[-1].binding != admission.binding
        or any(
            segment.idempotency_key != admission.idempotency.key
            or segment.request_digest != admission.idempotency.request_digest
            for segment in segments
        )
        or durable_tool_recovery_action(segments) is not admission.action
        or checkpoint.identity.conversation_id
        != execution.identity.conversation_id
        or checkpoint.identity.logical_turn_id
        != execution.identity.logical_turn_id
    ):
        raise ConversationConflictError()


def _outbox_target_matches(
    row: PgsqlRow | None,
    target: OutboxClaimTarget,
) -> bool:
    expected = (
        _row_str(row, "authority_digest")
        if row is not None
        else _CONCEALED_DIGEST
    )
    supplied = str(authority_digest(target.authority))
    return (
        supplied == expected
        and row is not None
        and _row_str(row, "intent_id") == target.intent_id
        and _row_str(row, "checkpoint_id") == str(target.checkpoint_id)
        and _row_str(row, "public_response_id")
        == str(target.public_response_id)
    )


def _outbox_row_to_record(
    row: PgsqlRow,
    outputs: tuple[ProviderLaneOutputCandidate, ...],
    *,
    state: OutboxState,
    attempts: int,
    lease_owner: str | None,
    lease_expires_at: datetime | None,
    published_at: datetime | None,
) -> OutboxRecord:
    return OutboxRecord(
        intent=PublicationIntent(
            intent_id=_row_str(row, "intent_id"),
            public_response_id=PublicResponseId(
                _row_str(row, "public_response_id")
            ),
            checkpoint_id=CheckpointId(_row_str(row, "checkpoint_id")),
            lane_outputs=tuple(value.public_output for value in outputs),
        ),
        authority_digest=AuthorityDigest(_row_str(row, "authority_digest")),
        state=state,
        attempts=attempts,
        lease_owner=lease_owner,
        lease_expires_at=lease_expires_at,
        published_at=published_at,
    )


def _row_str(row: PgsqlRow, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ConversationStorageError()
    return value


def _row_optional_str(row: PgsqlRow, key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ConversationStorageError()
    return value


def _row_int(row: PgsqlRow, key: str) -> int:
    value = row.get(key)
    if type(value) is not int or value < 0:
        raise ConversationStorageError()
    return value


def _row_optional_int(row: PgsqlRow, key: str) -> int | None:
    value = row.get(key)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ConversationStorageError()
    return value


def _row_bool(row: PgsqlRow, key: str) -> bool:
    value = row.get(key)
    if type(value) is not bool:
        raise ConversationStorageError()
    return value


def _row_datetime(row: PgsqlRow, key: str) -> datetime:
    value = row.get(key)
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ConversationStorageError()
    return value


def _row_optional_datetime(row: PgsqlRow, key: str) -> datetime | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ConversationStorageError()
    return value


def _row_bytes(row: PgsqlRow, key: str) -> bytes:
    value = row.get(key)
    if isinstance(value, memoryview):
        value = value.tobytes()
    if type(value) is not bytes or not value:
        raise ConversationStorageError()
    return value


def _validate_time(value: datetime) -> None:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ConversationValidationError()


_CHECK_SCHEMA_REVISION_SQL = f"""
SELECT "version_num"
FROM "avalan_task_alembic_version"
WHERE "version_num" = '{CONVERSATION_PGSQL_HEAD_REVISION}'
"""

_CHECK_SCHEMA_READINESS_SQL = """
SELECT
    "schema_version",
    "minimum_reader_version",
    "maximum_reader_version",
    "minimum_writer_version",
    "maximum_writer_version",
    "checkpoint_codec_version"
FROM "conversation_store_readiness"
"""

_LOCK_GLOBAL_CAPACITY_SQL = """
SELECT pg_advisory_xact_lock(736834691364134883)
"""

_INSERT_CONVERSATION_SQL = """
INSERT INTO "conversations" (
    "conversation_id", "authority_digest"
) VALUES (%s, %s)
ON CONFLICT ("conversation_id") DO NOTHING
"""

_SELECT_CONVERSATION_FOR_UPDATE_SQL = """
SELECT "authority_digest", "lifecycle_state"
FROM "conversations"
WHERE "conversation_id" = %s
FOR UPDATE
"""

_COUNT_CHECKPOINTS_SQL = """
SELECT COUNT(*)::BIGINT AS "record_count"
FROM "conversation_checkpoints"
WHERE "lifecycle_state" <> 'deleted'
  AND "checkpoint_id" NOT LIKE 'quarantine-%'
"""

_SELECT_CHECKPOINT_FOR_UPDATE_SQL = """
SELECT
    "checkpoint_id", "conversation_id", "authority_digest",
    "lifecycle_state", "checkpoint_sequence", "checkpoint_kind"
FROM "conversation_checkpoints"
WHERE "checkpoint_id" = %s
FOR UPDATE
"""

_COUNT_CHECKPOINT_CHILDREN_SQL = """
SELECT COUNT(*)::BIGINT AS "record_count"
FROM "conversation_checkpoints"
WHERE "parent_checkpoint_id" = %s
  AND "lifecycle_state" <> 'deleted'
"""

_INSERT_CHECKPOINT_SQL = """
INSERT INTO "conversation_checkpoints" (
    "checkpoint_id", "conversation_id", "logical_turn_id",
    "execution_segment_id", "branch_id", "parent_checkpoint_id",
    "checkpoint_sequence", "parent_sequence", "checkpoint_kind",
    "lifecycle_state", "authority_digest", "checkpoint_codec_version",
    "payload_schema_version", "payload_count", "payload_bytes",
    "lane_count", "provider_item_count", "opaque_byte_count",
    "created_at", "committed_at", "expires_at"
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
)
"""

_INSERT_LANE_SQL = """
INSERT INTO "conversation_lanes" (
    "checkpoint_id", "lane_id", "lane_sequence", "lane_mode",
    "binding_digest", "execution_digest", "provider_item_count",
    "opaque_byte_count", "upstream_deletion_state"
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

_INSERT_KEY_AUTHORITY_SQL = """
INSERT INTO "conversation_key_authorities" ("authority_digest")
VALUES (%s)
ON CONFLICT ("authority_digest") DO NOTHING
"""

_SELECT_KEY_AUTHORITY_FOR_UPDATE_SQL = """
SELECT
    "current_generation", "current_key_id", "current_key_revision"
FROM "conversation_key_authorities"
WHERE "authority_digest" = %s
FOR UPDATE
"""

_SELECT_KEY_REVISION_FOR_UPDATE_SQL = """
SELECT "key_status", "algorithm"
FROM "conversation_key_revisions"
WHERE "authority_digest" = %s
  AND "key_id" = %s
  AND "key_revision" = %s
FOR UPDATE
"""

_SELECT_KEY_GENERATION_FOR_UPDATE_SQL = """
SELECT "key_id"
FROM "conversation_key_revisions"
WHERE "authority_digest" = %s
  AND "key_revision" = %s
FOR UPDATE
"""

_DEMOTE_CURRENT_KEYS_SQL = """
UPDATE "conversation_key_revisions"
SET "key_status" = 'grace', "retired_at" = NULL
WHERE "authority_digest" = %s
  AND "key_status" = 'current'
  AND ("key_id", "key_revision") <> (%s, %s)
"""

_UPSERT_CURRENT_KEY_SQL = """
INSERT INTO "conversation_key_revisions" (
    "authority_digest", "key_id", "key_revision", "key_status", "algorithm"
) VALUES (%s, %s, %s, 'current', %s)
ON CONFLICT ("authority_digest", "key_id", "key_revision")
DO UPDATE SET "key_status" = 'current', "retired_at" = NULL
WHERE "conversation_key_revisions"."key_status" <> 'retired'
RETURNING "key_status", "algorithm"
"""

_UPDATE_KEY_AUTHORITY_SQL = """
UPDATE "conversation_key_authorities"
SET
    "current_generation" = %s,
    "current_key_id" = %s,
    "current_key_revision" = %s,
    "updated_at" = CURRENT_TIMESTAMP
WHERE "authority_digest" = %s
  AND "current_generation" = %s
RETURNING "current_generation"
"""

_INSERT_PAYLOAD_SQL = """
INSERT INTO "conversation_encrypted_payloads" (
    "payload_id", "authority_digest", "checkpoint_id", "conversation_id",
    "lane_id",
    "payload_sequence", "payload_kind", "payload_schema_version",
    "codec_version", "key_id", "key_revision", "algorithm", "nonce",
    "ciphertext", "authenticated_digest"
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
)
"""

_INSERT_PAYLOAD_REFERENCE_SQL = """
INSERT INTO "conversation_checkpoint_payload_refs" (
    "checkpoint_id", "conversation_id", "authority_digest", "lane_id",
    "payload_sequence", "payload_kind", "payload_schema_version",
    "codec_version", "key_id", "key_revision", "algorithm",
    "authenticated_digest", "payload_id"
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

_INSERT_CONTINUATION_REFERENCE_SQL = """
INSERT INTO "conversation_checkpoint_continuations" (
    "checkpoint_id", "conversation_id", "authority_digest",
    "execution_segment_id", "continuation_id",
    "continuation_state_revision", "continuation_digest",
    "definition_digest", "revision_binding_digest", "payload_lane_id",
    "payload_sequence", "payload_kind", "payload_id"
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

_SELECT_CHECKPOINT_PAYLOAD_SQL = """
SELECT
    p.*, k."key_status", c."lifecycle_state",
    c."execution_segment_id", c."checkpoint_sequence",
    c."conversation_id" AS "checkpoint_conversation_id",
    c."authority_digest" AS "checkpoint_authority_digest",
    c."payload_schema_version" AS "checkpoint_payload_schema_version",
    c."checkpoint_codec_version" AS "checkpoint_codec_version",
    r."checkpoint_id" AS "reference_checkpoint_id",
    r."conversation_id" AS "reference_conversation_id",
    r."authority_digest" AS "reference_authority_digest",
    r."lane_id" AS "reference_lane_id",
    r."payload_sequence" AS "reference_payload_sequence",
    r."payload_kind" AS "reference_payload_kind",
    r."payload_schema_version" AS "reference_payload_schema_version",
    r."codec_version" AS "reference_codec_version",
    r."key_id" AS "reference_key_id",
    r."key_revision" AS "reference_key_revision",
    r."algorithm" AS "reference_algorithm",
    r."authenticated_digest" AS "reference_authenticated_digest",
    r."payload_id" AS "reference_payload_id"
FROM "conversation_checkpoints" AS c
JOIN "conversation_checkpoint_payload_refs" AS r
  ON r."checkpoint_id" = c."checkpoint_id"
 AND r."payload_kind" = 'checkpoint'
JOIN "conversation_encrypted_payloads" AS p
  ON p."payload_id" = r."payload_id"
JOIN "conversation_key_revisions" AS k
  ON k."authority_digest" = p."authority_digest"
 AND k."key_id" = p."key_id"
 AND k."key_revision" = p."key_revision"
WHERE c."checkpoint_id" = %s
  AND c."authority_digest" = %s
"""

_SELECT_OUTPUT_PAYLOADS_SQL = """
SELECT
    p.*, k."key_status",
    c."conversation_id" AS "checkpoint_conversation_id",
    c."authority_digest" AS "checkpoint_authority_digest",
    c."payload_schema_version" AS "checkpoint_payload_schema_version",
    c."checkpoint_codec_version" AS "checkpoint_codec_version",
    r."checkpoint_id" AS "reference_checkpoint_id",
    r."conversation_id" AS "reference_conversation_id",
    r."authority_digest" AS "reference_authority_digest",
    r."lane_id" AS "reference_lane_id",
    r."payload_sequence" AS "reference_payload_sequence",
    r."payload_kind" AS "reference_payload_kind",
    r."payload_schema_version" AS "reference_payload_schema_version",
    r."codec_version" AS "reference_codec_version",
    r."key_id" AS "reference_key_id",
    r."key_revision" AS "reference_key_revision",
    r."algorithm" AS "reference_algorithm",
    r."authenticated_digest" AS "reference_authenticated_digest",
    r."payload_id" AS "reference_payload_id",
    l."lane_id" AS "registered_lane_id"
FROM "conversation_checkpoints" AS c
JOIN "conversation_checkpoint_payload_refs" AS r
  ON r."checkpoint_id" = c."checkpoint_id"
 AND r."payload_kind" = 'lane_output'
JOIN "conversation_encrypted_payloads" AS p
  ON p."payload_id" = r."payload_id"
JOIN "conversation_lanes" AS l
  ON l."checkpoint_id" = c."checkpoint_id"
 AND l."lane_id" = r."lane_id"
JOIN "conversation_key_revisions" AS k
  ON k."authority_digest" = p."authority_digest"
 AND k."key_id" = p."key_id"
 AND k."key_revision" = p."key_revision"
WHERE c."checkpoint_id" = %s
  AND c."authority_digest" = %s
  AND c."lifecycle_state" = 'committed'
ORDER BY r."payload_sequence", r."lane_id"
"""

_SELECT_CONTINUATION_PAYLOAD_SQL = """
SELECT
    p.*, k."key_status", x."continuation_id",
    x."continuation_state_revision", x."continuation_digest",
    x."definition_digest", x."revision_binding_digest",
    c."conversation_id" AS "checkpoint_conversation_id",
    c."authority_digest" AS "checkpoint_authority_digest",
    c."payload_schema_version" AS "checkpoint_payload_schema_version",
    c."checkpoint_codec_version" AS "checkpoint_codec_version",
    r."checkpoint_id" AS "reference_checkpoint_id",
    r."conversation_id" AS "reference_conversation_id",
    r."authority_digest" AS "reference_authority_digest",
    r."lane_id" AS "reference_lane_id",
    r."payload_sequence" AS "reference_payload_sequence",
    r."payload_kind" AS "reference_payload_kind",
    r."payload_schema_version" AS "reference_payload_schema_version",
    r."codec_version" AS "reference_codec_version",
    r."key_id" AS "reference_key_id",
    r."key_revision" AS "reference_key_revision",
    r."algorithm" AS "reference_algorithm",
    r."authenticated_digest" AS "reference_authenticated_digest",
    r."payload_id" AS "reference_payload_id",
    x."conversation_id" AS "continuation_conversation_id",
    x."authority_digest" AS "continuation_authority_digest",
    x."payload_lane_id" AS "continuation_payload_lane_id",
    x."payload_sequence" AS "continuation_payload_sequence",
    x."payload_kind" AS "continuation_payload_kind"
FROM "conversation_checkpoints" AS c
JOIN "conversation_checkpoint_continuations" AS x
  ON x."checkpoint_id" = c."checkpoint_id"
JOIN "conversation_checkpoint_payload_refs" AS r
  ON r."checkpoint_id" = x."checkpoint_id"
 AND r."payload_id" = x."payload_id"
JOIN "conversation_encrypted_payloads" AS p
  ON p."payload_id" = x."payload_id"
JOIN "conversation_key_revisions" AS k
  ON k."authority_digest" = p."authority_digest"
 AND k."key_id" = p."key_id"
 AND k."key_revision" = p."key_revision"
WHERE c."checkpoint_id" = %s
  AND c."authority_digest" = %s
  AND c."lifecycle_state" = 'committed'
"""

_SELECT_IDEMPOTENCY_FOR_UPDATE_SQL = """
SELECT *
FROM "conversation_idempotency"
WHERE "authority_digest" = %s
  AND "operation" = %s
  AND "idempotency_key" = %s
FOR UPDATE
"""

_SELECT_IDEMPOTENCY_SQL = """
SELECT *
FROM "conversation_idempotency"
WHERE "authority_digest" = %s
  AND "operation" = %s
  AND "idempotency_key" = %s
"""

_COUNT_IDEMPOTENCY_SQL = """
SELECT COUNT(*)::BIGINT AS "record_count"
FROM "conversation_idempotency"
"""

_COUNT_IDEMPOTENCY_IN_FLIGHT_SQL = """
SELECT COUNT(*)::BIGINT AS "record_count"
FROM "conversation_idempotency"
WHERE "record_state" IN ('in_progress', 'ambiguous')
"""

_INSERT_IDEMPOTENCY_SQL = """
INSERT INTO "conversation_idempotency" (
    "authority_digest", "operation", "idempotency_key", "request_digest",
    "record_state", "owner_token", "lease_expires_at", "execution_digest",
    "created_at", "updated_at"
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

_UPDATE_IDEMPOTENCY_STATE_SQL = """
UPDATE "conversation_idempotency"
SET "record_state" = %s, "updated_at" = %s
WHERE "authority_digest" = %s
  AND "operation" = %s
  AND "idempotency_key" = %s
"""

_UPDATE_IDEMPOTENCY_RECOVERY_LEASE_SQL = """
UPDATE "conversation_idempotency"
SET "record_state" = %s,
    "owner_token" = %s,
    "lease_expires_at" = %s,
    "updated_at" = %s
WHERE "authority_digest" = %s
  AND "operation" = %s
  AND "idempotency_key" = %s
"""

_DELETE_IDEMPOTENCY_SQL = """
DELETE FROM "conversation_idempotency"
WHERE "authority_digest" = %s
  AND "operation" = %s
  AND "idempotency_key" = %s
"""

_INSERT_EXECUTION_RESERVATION_LANE_SQL = """
INSERT INTO "conversation_execution_reservation_lanes" (
    "authority_digest", "operation", "idempotency_key", "checkpoint_id",
    "conversation_id", "logical_turn_id", "execution_segment_id",
    "branch_id", "checkpoint_sequence", "parent_checkpoint_id",
    "parent_sequence", "lane_id", "binding_digest", "lane_mode",
    "output_scope"
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
)
"""

_SELECT_EXECUTION_RESERVATION_LANE_SQL = """
SELECT *
FROM "conversation_execution_reservation_lanes"
WHERE "authority_digest" = %s
  AND "operation" = %s
  AND "idempotency_key" = %s
  AND "lane_id" = %s
"""

_COUNT_EXECUTION_STAGING_SQL = """
SELECT COUNT(*)::BIGINT AS "record_count"
FROM "conversation_execution_staging"
"""

_INSERT_EXECUTION_STAGING_SQL = """
INSERT INTO "conversation_execution_staging" (
    "staging_id", "authority_digest", "operation", "idempotency_key",
    "request_digest", "owner_token", "checkpoint_id", "lane_id",
    "binding_digest", "execution_digest", "lane_mode", "output_scope",
    "item_count", "opaque_byte_count"
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

_SELECT_EXECUTION_STAGING_OWNER_SQL = """
SELECT *
FROM "conversation_execution_staging"
WHERE "owner_token" = %s
  AND "checkpoint_id" = %s
ORDER BY "lane_id"
FOR UPDATE
"""

_DELETE_EXECUTION_STAGING_OWNER_SQL = """
DELETE FROM "conversation_execution_staging"
WHERE "owner_token" = %s
  AND "checkpoint_id" = %s
"""

_DELETE_EXECUTION_STAGING_ALL_OWNER_SQL = """
DELETE FROM "conversation_execution_staging"
WHERE "owner_token" = %s
"""

_SELECT_IDEMPOTENCY_BY_OWNER_FOR_UPDATE_SQL = """
SELECT *
FROM "conversation_idempotency"
WHERE "owner_token" = %s
FOR UPDATE
"""

_COUNT_RESPONSE_ALLOCATIONS_SQL = """
SELECT
    (SELECT COUNT(*) FROM "conversation_provisional_responses")::BIGINT
        AS "provisional_count",
    (
        (SELECT COUNT(*) FROM "conversation_provisional_responses")
        + (SELECT COUNT(*) FROM "conversation_public_responses")
    )::BIGINT AS "total_count"
"""

_INSERT_PROVISIONAL_SQL = """
INSERT INTO "conversation_provisional_responses" (
    "provisional_response_id", "public_response_id", "owner_token",
    "authority_digest"
) VALUES (%s, %s, %s, %s)
"""

_SELECT_PROVISIONAL_FOR_UPDATE_SQL = """
SELECT *
FROM "conversation_provisional_responses"
WHERE "provisional_response_id" = %s
FOR UPDATE
"""

_SELECT_PROVISIONAL_BY_OWNER_SQL = """
SELECT "owner_token"
FROM "conversation_provisional_responses"
WHERE "owner_token" = %s
LIMIT 1
"""

_DELETE_PROVISIONAL_SQL = """
DELETE FROM "conversation_provisional_responses"
WHERE "provisional_response_id" = %s
"""

_DELETE_PROVISIONAL_OWNER_SQL = """
DELETE FROM "conversation_provisional_responses"
WHERE "owner_token" = %s
"""

_INSERT_PUBLIC_RESPONSE_SQL = """
INSERT INTO "conversation_public_responses" (
    "public_response_id", "checkpoint_id", "authority_digest"
) VALUES (%s, %s, %s)
"""

_SELECT_PUBLIC_RESPONSE_SQL = """
SELECT "public_response_id", "checkpoint_id", "authority_digest", "tombstoned"
FROM "conversation_public_responses"
WHERE "public_response_id" = %s
  AND "authority_digest" = %s
"""

_SELECT_PUBLIC_RESPONSE_FOR_UPDATE_SQL = """
SELECT "public_response_id", "checkpoint_id", "authority_digest", "tombstoned"
FROM "conversation_public_responses"
WHERE "public_response_id" = %s
FOR UPDATE
"""

_INSERT_HEAD_SQL = """
INSERT INTO "conversation_named_heads" (
    "authority_digest", "head_id", "head_revision", "checkpoint_id",
    "lifecycle_state"
) VALUES (%s, %s, %s, %s, %s)
"""

_COUNT_HEADS_SQL = """
SELECT COUNT(*)::BIGINT AS "record_count"
FROM "conversation_named_heads"
"""

_SELECT_HEAD_SQL = """
SELECT "head_revision", "checkpoint_id", "lifecycle_state"
FROM "conversation_named_heads"
WHERE "authority_digest" = %s AND "head_id" = %s
"""

_SELECT_HEAD_FOR_UPDATE_SQL = """
SELECT "head_revision", "checkpoint_id", "lifecycle_state"
FROM "conversation_named_heads"
WHERE "authority_digest" = %s AND "head_id" = %s
FOR UPDATE
"""

_UPDATE_HEAD_SQL = """
UPDATE "conversation_named_heads"
SET
    "checkpoint_id" = %s,
    "head_revision" = "head_revision" + 1,
    "updated_at" = %s
WHERE "authority_digest" = %s
  AND "head_id" = %s
  AND "head_revision" = %s
  AND "checkpoint_id" = %s
  AND "lifecycle_state" = 'active'
"""

_INSERT_OUTBOX_SQL = """
INSERT INTO "conversation_outbox" (
    "intent_id", "checkpoint_id", "public_response_id", "authority_digest"
) VALUES (%s, %s, %s, %s)
"""

_COUNT_OUTBOX_SQL = """
SELECT COUNT(*)::BIGINT AS "record_count"
FROM "conversation_outbox"
"""

_COMMIT_IDEMPOTENCY_SQL = """
UPDATE "conversation_idempotency"
SET
    "record_state" = 'committed',
    "checkpoint_id" = %s,
    "public_response_id" = %s,
    "updated_at" = %s
WHERE "authority_digest" = %s
  AND "operation" = %s
  AND "idempotency_key" = %s
  AND "owner_token" = %s
  AND "request_digest" = %s
  AND "record_state" = 'in_progress'
"""

_COUNT_AUTHORIZED_CHILDREN_SQL = """
SELECT
    (
        SELECT COUNT(*)
        FROM "conversation_checkpoints"
        WHERE "checkpoint_id" = %s
          AND "authority_digest" = %s
          AND "lifecycle_state" = 'committed'
    )::BIGINT AS "parent_count",
    (
        SELECT COUNT(*)
        FROM "conversation_checkpoints"
        WHERE "parent_checkpoint_id" = %s
          AND "lifecycle_state" <> 'deleted'
    )::BIGINT AS "child_count"
"""

_DELETE_ENVELOPE_REFERENCE_SQL = """
DELETE FROM "conversation_checkpoint_payload_refs"
WHERE "checkpoint_id" = %s AND "payload_kind" = 'checkpoint'
"""

_TOMBSTONE_CHECKPOINT_SQL = """
UPDATE "conversation_checkpoints"
SET "lifecycle_state" = 'tombstoned', "tombstoned_at" = %s
WHERE "checkpoint_id" = %s
  AND "authority_digest" = %s
  AND "lifecycle_state" = 'committed'
"""

_TOMBSTONE_PUBLIC_RESPONSE_SQL = """
UPDATE "conversation_public_responses"
SET "tombstoned" = TRUE, "tombstoned_at" = %s
WHERE "public_response_id" = %s
  AND "authority_digest" = %s
  AND NOT "tombstoned"
"""

_DELETE_OUTBOX_CHECKPOINT_SQL = """
DELETE FROM "conversation_outbox" WHERE "checkpoint_id" = %s
"""

_TOMBSTONE_HEADS_SQL = """
UPDATE "conversation_named_heads"
SET "lifecycle_state" = 'tombstoned', "updated_at" = %s
WHERE "checkpoint_id" = %s
"""

_SELECT_STORED_LANES_SQL = """
SELECT
    l."lane_id", c."conversation_id", r."payload_id"
FROM "conversation_lanes" AS l
JOIN "conversation_checkpoints" AS c
  ON c."checkpoint_id" = l."checkpoint_id"
JOIN "conversation_checkpoint_payload_refs" AS r
  ON r."checkpoint_id" = l."checkpoint_id"
 AND r."lane_id" = l."lane_id"
 AND r."payload_kind" = 'deletion_target'
WHERE l."checkpoint_id" = %s AND l."lane_mode" = 'stored'
ORDER BY l."lane_sequence"
"""

_SELECT_STORED_LANES_FOR_DELETE_SQL = """
SELECT "lane_id", "upstream_deletion_state"
FROM "conversation_lanes"
WHERE "checkpoint_id" = %s AND "lane_mode" = 'stored'
ORDER BY "lane_sequence"
FOR UPDATE
"""

_INSERT_RECONCILIATION_SQL = """
INSERT INTO "conversation_reconciliation_outbox" (
    "reconciliation_id", "checkpoint_id", "lane_id", "authority_digest",
    "target_conversation_id", "target_payload_sequence",
    "target_payload_kind", "target_payload_id", "work_kind"
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT ("checkpoint_id", "lane_id", "work_kind") DO NOTHING
"""

_DELETE_ALL_PAYLOAD_REFERENCES_SQL = """
DELETE FROM "conversation_checkpoint_payload_refs" WHERE "checkpoint_id" = %s
"""

_DELETE_IDEMPOTENCY_CHECKPOINT_SQL = """
DELETE FROM "conversation_idempotency" WHERE "checkpoint_id" = %s
"""

_DELETE_PUBLIC_RESPONSE_SQL = """
DELETE FROM "conversation_public_responses"
WHERE "public_response_id" = %s AND "authority_digest" = %s
"""

_DELETE_CHECKPOINT_LOGICALLY_SQL = """
UPDATE "conversation_checkpoints"
SET "lifecycle_state" = 'deleted', "deleted_at" = %s
WHERE "checkpoint_id" = %s
  AND "authority_digest" = %s
  AND "lifecycle_state" = 'tombstoned'
"""

_LIST_CHECKPOINTS_SQL = """
SELECT "checkpoint_id"
FROM "conversation_checkpoints"
WHERE "authority_digest" = %s
  AND "lifecycle_state" = 'committed'
  AND "checkpoint_id" > %s
ORDER BY "checkpoint_id"
LIMIT %s
"""

_SELECT_EXPIRED_FOR_DELETE_SQL = """
SELECT "checkpoint_id"
FROM "conversation_checkpoints"
WHERE "lifecycle_state" = 'expired'
  AND NOT EXISTS (
      SELECT 1
      FROM "conversation_lanes" AS l
      WHERE l."checkpoint_id" = "conversation_checkpoints"."checkpoint_id"
        AND l."lane_mode" = 'stored'
        AND l."upstream_deletion_state" NOT IN ('succeeded', 'unsupported')
  )
ORDER BY "checkpoint_id"
LIMIT %s
FOR UPDATE SKIP LOCKED
"""

_SELECT_COMMITTED_EXPIRY_SQL = """
SELECT "checkpoint_id", "authority_digest"
FROM "conversation_checkpoints"
WHERE "lifecycle_state" = 'committed'
  AND "expires_at" IS NOT NULL
  AND "expires_at" <= %s
ORDER BY "expires_at", "checkpoint_id"
LIMIT %s
FOR UPDATE SKIP LOCKED
"""

_MARK_CHECKPOINT_EXPIRED_SQL = """
UPDATE "conversation_checkpoints"
SET "lifecycle_state" = 'expired'
WHERE "checkpoint_id" = %s AND "lifecycle_state" = 'committed'
"""

_EXPIRE_PUBLIC_RESPONSE_SQL = """
UPDATE "conversation_public_responses"
SET "tombstoned" = TRUE, "tombstoned_at" = %s
WHERE "checkpoint_id" = %s AND NOT "tombstoned"
"""

_DELETE_PUBLIC_RESPONSE_CHECKPOINT_SQL = """
DELETE FROM "conversation_public_responses" WHERE "checkpoint_id" = %s
"""

_DELETE_EXPIRED_CHECKPOINT_SQL = """
UPDATE "conversation_checkpoints"
SET "lifecycle_state" = 'deleted', "deleted_at" = %s
WHERE "checkpoint_id" = %s AND "lifecycle_state" = 'expired'
"""

_COUNT_TERMINAL_SQL = """
SELECT COUNT(*)::BIGINT AS "record_count"
FROM "conversation_terminal_metadata"
"""

_SELECT_DELETION_TERMINAL_SQL = """
SELECT t."terminal_state"
FROM "conversation_terminal_metadata" AS t
JOIN "conversation_checkpoints" AS c
  ON c."checkpoint_id" = t."checkpoint_id"
WHERE t."public_response_id" = %s
  AND c."authority_digest" = %s
"""

_DELETE_OLDEST_TERMINAL_SQL = """
DELETE FROM "conversation_terminal_metadata"
WHERE "checkpoint_id" = (
    SELECT "checkpoint_id"
    FROM "conversation_terminal_metadata"
    ORDER BY "terminal_at", "checkpoint_id"
    LIMIT 1
)
"""

_UPSERT_TERMINAL_SQL = """
INSERT INTO "conversation_terminal_metadata" (
    "checkpoint_id", "public_response_id", "terminal_state", "terminal_at"
) VALUES (%s, %s, %s, %s)
ON CONFLICT ("checkpoint_id") DO UPDATE SET
    "public_response_id" = EXCLUDED."public_response_id",
    "terminal_state" = EXCLUDED."terminal_state",
    "terminal_at" = EXCLUDED."terminal_at"
"""

_SELECT_PUBLISHED_OUTBOX_PRUNE_SQL = """
SELECT "intent_id"
FROM "conversation_outbox"
WHERE "outbox_state" = 'published' AND "published_at" <= %s
ORDER BY "published_at", "intent_id"
LIMIT %s
FOR UPDATE SKIP LOCKED
"""

_DELETE_OUTBOX_SQL = """
DELETE FROM "conversation_outbox" WHERE "intent_id" = %s
"""

_SELECT_IDEMPOTENCY_PRUNE_SQL = """
SELECT "authority_digest", "operation", "idempotency_key"
FROM "conversation_idempotency" AS i
WHERE i."record_state" = 'failed_no_dispatch'
   OR (
       i."record_state" = 'committed'
       AND NOT EXISTS (
           SELECT 1 FROM "conversation_checkpoints" AS c
           WHERE c."checkpoint_id" = i."checkpoint_id"
             AND c."lifecycle_state" <> 'deleted'
       )
   )
ORDER BY i."updated_at", i."idempotency_key"
LIMIT %s
FOR UPDATE SKIP LOCKED
"""

_SELECT_OUTBOX_FOR_UPDATE_SQL = """
SELECT * FROM "conversation_outbox" WHERE "intent_id" = %s FOR UPDATE
"""

_SELECT_OUTBOX_SQL = """
SELECT * FROM "conversation_outbox" WHERE "intent_id" = %s
"""

_CLAIM_OUTBOX_SQL = """
UPDATE "conversation_outbox"
SET
    "outbox_state" = 'claimed',
    "attempts" = "attempts" + 1,
    "lease_owner" = %s,
    "lease_expires_at" = %s,
    "published_at" = NULL
WHERE "intent_id" = %s
  AND "outbox_state" IN ('pending', 'claimed')
"""

_SELECT_RECOVERABLE_OUTBOX_SQL = """
SELECT *
FROM "conversation_outbox"
WHERE "authority_digest" = %s
  AND (
      "outbox_state" = 'pending'
      OR (
          "outbox_state" = 'claimed'
          AND "lease_expires_at" <= %s
      )
  )
ORDER BY "available_sequence", "intent_id"
LIMIT %s
FOR UPDATE SKIP LOCKED
"""

_ACKNOWLEDGE_OUTBOX_SQL = """
UPDATE "conversation_outbox"
SET
    "outbox_state" = 'published',
    "lease_owner" = NULL,
    "lease_expires_at" = NULL,
    "published_at" = %s
WHERE "intent_id" = %s
  AND "lease_owner" = %s
  AND "outbox_state" = 'claimed'
"""

_RELEASE_OUTBOX_SQL = """
UPDATE "conversation_outbox"
SET
    "outbox_state" = 'pending',
    "lease_owner" = NULL,
    "lease_expires_at" = NULL,
    "available_sequence" = nextval(
        pg_get_serial_sequence('conversation_outbox', 'available_sequence')
    )
WHERE "intent_id" = %s
  AND "lease_owner" = %s
  AND "outbox_state" = 'claimed'
"""

_SELECT_RECONCILIATION_SQL = """
SELECT
    o.*, p.*, k."key_status",
    l."binding_digest",
    c."conversation_id" AS "checkpoint_conversation_id",
    c."authority_digest" AS "checkpoint_authority_digest",
    c."payload_schema_version" AS "checkpoint_payload_schema_version",
    c."checkpoint_codec_version" AS "checkpoint_codec_version",
    c."lifecycle_state" AS "checkpoint_lifecycle_state",
    r."checkpoint_id" AS "reference_checkpoint_id",
    r."conversation_id" AS "reference_conversation_id",
    r."authority_digest" AS "reference_authority_digest",
    r."lane_id" AS "reference_lane_id",
    r."payload_sequence" AS "reference_payload_sequence",
    r."payload_kind" AS "reference_payload_kind",
    r."payload_schema_version" AS "reference_payload_schema_version",
    r."codec_version" AS "reference_codec_version",
    r."key_id" AS "reference_key_id",
    r."key_revision" AS "reference_key_revision",
    r."algorithm" AS "reference_algorithm",
    r."authenticated_digest" AS "reference_authenticated_digest",
    r."payload_id" AS "reference_payload_id"
FROM "conversation_reconciliation_outbox" AS o
JOIN "conversation_checkpoints" AS c
  ON c."checkpoint_id" = o."checkpoint_id"
 AND c."conversation_id" = o."target_conversation_id"
 AND c."authority_digest" = o."authority_digest"
JOIN "conversation_lanes" AS l
  ON l."checkpoint_id" = o."checkpoint_id"
 AND l."lane_id" = o."lane_id"
JOIN "conversation_checkpoint_payload_refs" AS r
  ON r."checkpoint_id" = o."checkpoint_id"
 AND r."conversation_id" = o."target_conversation_id"
 AND r."authority_digest" = o."authority_digest"
 AND r."lane_id" = o."lane_id"
 AND r."payload_sequence" = o."target_payload_sequence"
 AND r."payload_kind" = o."target_payload_kind"
 AND r."payload_id" = o."target_payload_id"
JOIN "conversation_encrypted_payloads" AS p
  ON p."payload_id" = r."payload_id"
JOIN "conversation_key_revisions" AS k
  ON k."authority_digest" = p."authority_digest"
 AND k."key_id" = p."key_id"
 AND k."key_revision" = p."key_revision"
WHERE o."authority_digest" = %s
  AND (
      o."work_state" IN ('pending', 'failed')
      OR (o."work_state" = 'claimed' AND o."lease_expires_at" <= %s)
  )
ORDER BY o."created_at", o."reconciliation_id"
LIMIT %s
FOR UPDATE OF o SKIP LOCKED
"""

_SELECT_PROVIDER_LIFECYCLE_SQL = _SELECT_RECONCILIATION_SQL.replace(
    'WHERE o."authority_digest" = %s',
    "WHERE o.\"work_kind\" = 'delete_upstream'\n"
    '  AND o."authority_digest" = %s',
)

_CLAIM_RECONCILIATION_SQL = """
UPDATE "conversation_reconciliation_outbox"
SET
    "work_state" = 'claimed',
    "attempts" = "attempts" + 1,
    "lease_owner" = %s,
    "lease_expires_at" = %s,
    "completed_at" = NULL
WHERE "reconciliation_id" = %s
"""

_SELECT_RECONCILIATION_FOR_UPDATE_SQL = """
SELECT
    o.*, p.*, k."key_status",
    l."binding_digest",
    c."conversation_id" AS "checkpoint_conversation_id",
    c."authority_digest" AS "checkpoint_authority_digest",
    c."payload_schema_version" AS "checkpoint_payload_schema_version",
    c."checkpoint_codec_version" AS "checkpoint_codec_version",
    c."lifecycle_state" AS "checkpoint_lifecycle_state",
    r."checkpoint_id" AS "reference_checkpoint_id",
    r."conversation_id" AS "reference_conversation_id",
    r."authority_digest" AS "reference_authority_digest",
    r."lane_id" AS "reference_lane_id",
    r."payload_sequence" AS "reference_payload_sequence",
    r."payload_kind" AS "reference_payload_kind",
    r."payload_schema_version" AS "reference_payload_schema_version",
    r."codec_version" AS "reference_codec_version",
    r."key_id" AS "reference_key_id",
    r."key_revision" AS "reference_key_revision",
    r."algorithm" AS "reference_algorithm",
    r."authenticated_digest" AS "reference_authenticated_digest",
    r."payload_id" AS "reference_payload_id"
FROM "conversation_reconciliation_outbox" AS o
JOIN "conversation_checkpoints" AS c
  ON c."checkpoint_id" = o."checkpoint_id"
 AND c."conversation_id" = o."target_conversation_id"
 AND c."authority_digest" = o."authority_digest"
JOIN "conversation_lanes" AS l
  ON l."checkpoint_id" = o."checkpoint_id"
 AND l."lane_id" = o."lane_id"
JOIN "conversation_checkpoint_payload_refs" AS r
  ON r."checkpoint_id" = o."checkpoint_id"
 AND r."conversation_id" = o."target_conversation_id"
 AND r."authority_digest" = o."authority_digest"
 AND r."lane_id" = o."lane_id"
 AND r."payload_sequence" = o."target_payload_sequence"
 AND r."payload_kind" = o."target_payload_kind"
 AND r."payload_id" = o."target_payload_id"
JOIN "conversation_encrypted_payloads" AS p
  ON p."payload_id" = r."payload_id"
JOIN "conversation_key_revisions" AS k
  ON k."authority_digest" = p."authority_digest"
 AND k."key_id" = p."key_id"
 AND k."key_revision" = p."key_revision"
WHERE o."reconciliation_id" = %s
FOR UPDATE OF o
"""

_ACKNOWLEDGE_RECONCILIATION_SQL = """
UPDATE "conversation_reconciliation_outbox"
SET
    "work_state" = %s,
    "lease_owner" = NULL,
    "lease_expires_at" = NULL,
    "completed_at" = %s
WHERE "reconciliation_id" = %s
  AND "lease_owner" = %s
  AND "work_state" = 'claimed'
"""

_UPDATE_LANE_RECONCILIATION_SQL = """
UPDATE "conversation_lanes"
SET "upstream_deletion_state" = %s
WHERE "checkpoint_id" = %s AND "lane_id" = %s
"""

_SELECT_GARBAGE_PAYLOADS_SQL = """
SELECT "payload_id"
FROM "conversation_encrypted_payloads"
WHERE "reference_count" = 0
ORDER BY "created_at", "payload_id"
LIMIT %s
FOR UPDATE SKIP LOCKED
"""

_DELETE_GARBAGE_PAYLOAD_SQL = """
DELETE FROM "conversation_encrypted_payloads"
WHERE "payload_id" = %s AND "reference_count" = 0
"""

_SELECT_KEY_ROTATION_PAYLOADS_SQL = """
SELECT p.*, k."key_status"
FROM "conversation_encrypted_payloads" AS p
JOIN "conversation_key_revisions" AS k
  ON k."authority_digest" = p."authority_digest"
 AND k."key_id" = p."key_id"
 AND k."key_revision" = p."key_revision"
WHERE p."authority_digest" = %s
  AND (p."key_id", p."key_revision") <> (%s, %s)
  AND p."reference_count" = 1
  AND k."key_status" IN ('current', 'grace')
ORDER BY p."created_at", p."payload_id"
LIMIT %s
"""

_SELECT_PAYLOAD_FOR_UPDATE_SQL = """
SELECT *
FROM "conversation_encrypted_payloads"
WHERE "payload_id" = %s
FOR UPDATE
"""

_UPDATE_ROTATED_PAYLOAD_SQL = """
UPDATE "conversation_encrypted_payloads"
SET
    "key_id" = %s,
    "key_revision" = %s,
    "algorithm" = %s,
    "nonce" = %s,
    "ciphertext" = %s,
    "authenticated_digest" = %s
WHERE "payload_id" = %s
  AND "key_id" = %s
  AND "key_revision" = %s
  AND "authenticated_digest" = %s
"""

_COUNT_KEY_PAYLOAD_REFERENCES_SQL = """
SELECT COUNT(*)::BIGINT AS "record_count"
FROM "conversation_encrypted_payloads"
WHERE "authority_digest" = %s
  AND "key_id" = %s
  AND "key_revision" = %s
  AND "reference_count" > 0
"""

_RETIRE_KEY_SQL = """
UPDATE "conversation_key_revisions"
SET "key_status" = 'retired', "retired_at" = %s
WHERE "authority_digest" = %s
  AND "key_id" = %s
  AND "key_revision" = %s
  AND "key_status" = 'grace'
"""
