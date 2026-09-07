"""Prepare validated encrypted trigger inputs outside admission locks."""

from ..task.artifact_ownership import ArtifactObject, ArtifactStagingOwner
from ..task.artifact_staging import StagedArtifactStore
from ..task.artifacts.object_store import ObjectArtifactStore
from ..task.artifacts.ownership_memory import MemoryArtifactOwnership
from ..task.artifacts.ownership_pgsql import PgsqlArtifactOwnership
from ..task.artifacts.pgsql import PgsqlArtifactStore
from ..task.client import TaskClient
from ..task.definition import TaskDefinition
from ..task.materialization import task_file_descriptors_from_input
from ..task.preparation import (
    PreparedTaskInput,
    prepare_task_input,
    restore_task_input,
)
from ..task.privacy import DecryptionProvider
from ..task.submission import (
    PreparedTaskSubmission,
    TaskSubmissionOutcome,
    TaskSubmissionRequest,
)
from ..task.validation import TaskValidationError, TaskValidationIssue
from .admission import (
    PreparedTriggerAdmission,
    TriggerAdmissionCancelledError,
    TriggerAdmissionResult,
    TriggerCommitOutcome,
)
from .bindings import bind_input, bind_values
from .canonical import configuration_hash, plain_value
from .definition import CronTrigger, TriggerConfiguration
from .error import TriggerError, TriggerErrorCode
from .plan import PlannedOccurrence, TriggerAdmissionPlan
from .records import OwnerScopeId
from .registration import TriggerRegistration
from .schedule import ScheduleProvenance, schedule_provenance
from .sealed_input import RegisteredTriggerInput, seal_input, unseal_input
from .stores.memory_admission import MemoryTriggerAdmissionStore
from .stores.pgsql_admission import PgsqlTriggerAdmissionStore

from asyncio import CancelledError
from dataclasses import dataclass, field, replace
from uuid import uuid4

_REGISTRATION_AUTHORITY = object()


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedTriggerRegistration:
    """Retain real input preparation and recoverable staging identity."""

    registration: TriggerRegistration
    task_input: PreparedTaskInput = field(repr=False)
    staging: ArtifactStagingOwner
    objects: tuple[ArtifactObject, ...] = field(repr=False)
    _authority: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        assert self._authority is _REGISTRATION_AUTHORITY
        assert (
            self.registration.task_definition_id
            == self.task_input.definition_id
        )
        client = self.task_input._client
        assert (
            self.registration.execution_deployment_id
            == client._execution_deployment_id
        )
        assert self.staging.owner_scope == client._owner_scope
        assert self.objects == tuple(
            ArtifactObject.from_ref(value.ref)
            for value in self.task_input.materialized_files
        )


class TriggerPreparationFailure(TriggerError):
    """Retain resources when external preparation did not settle safely."""

    def __init__(
        self,
        *,
        error: Exception | None = None,
        registration: PreparedTriggerRegistration | None = None,
        submissions: tuple[PreparedTaskSubmission, ...] = (),
        staging: ArtifactStagingOwner | None = None,
        objects: tuple[ArtifactObject, ...] = (),
    ) -> None:
        self.staging = staging
        self.objects = objects
        self.registration = registration
        self.submissions = submissions
        self.admission_outcome = TriggerCommitOutcome.NOT_COMMITTED
        self.resources_uncertain = bool(objects or submissions or registration)
        self.validation_issues: tuple[TaskValidationIssue, ...] = (
            error.issues if isinstance(error, TaskValidationError) else ()
        )
        self.failure = error
        if isinstance(error, TriggerError):
            code, path = error.code, error.path
        elif isinstance(error, TaskValidationError):
            code, path = TriggerErrorCode.INVALID_CONFIG, "input.task"
        else:
            code, path = TriggerErrorCode.ADMISSION_RETRYABLE, "preparation"
        super().__init__(code, path)


class TriggerPreparationCancelledError(CancelledError):
    """Retain staged resource identities while preserving cancellation."""

    def __init__(self, preparation: TriggerPreparationFailure) -> None:
        self.preparation = preparation
        super().__init__("trigger preparation cancelled")


class TriggerPreparationService:
    """Reuse the task client's validation with concrete store capabilities.

    The configured deployment ID is an opaque host identity. This service
    does not attest a complete execution deployment closure.
    """

    def __init__(
        self,
        client: TaskClient,
        admission: PgsqlTriggerAdmissionStore | MemoryTriggerAdmissionStore,
        ownership: PgsqlArtifactOwnership | MemoryArtifactOwnership,
        decryption: DecryptionProvider,
    ) -> None:
        self.client = client
        self.admission = admission
        self.ownership = ownership
        self.decryption = decryption
        self.owner = OwnerScopeId(value=client._owner_scope)
        if self.owner != admission.store.owner:
            raise TriggerError(
                TriggerErrorCode.STORE_INCOMPATIBLE, "owner_scope_id"
            )
        if isinstance(admission, PgsqlTriggerAdmissionStore):
            if (
                client._queue is not admission.queue
                or not isinstance(ownership, PgsqlArtifactOwnership)
                or ownership.database is not admission.store.database
            ):
                raise TriggerError(
                    TriggerErrorCode.STORE_INCOMPATIBLE, "store.database"
                )
        elif (
            client._queue is not admission.participant
            or ownership is not admission.ownership
        ):
            raise TriggerError(
                TriggerErrorCode.STORE_INCOMPATIBLE, "store.memory"
            )

    def _durable_input(
        self, definition: TaskDefinition, value: object
    ) -> None:
        files = task_file_descriptors_from_input(definition, value)
        local = False
        for descriptor in files:
            provider = descriptor.provider_reference
            if provider is not None:
                if not provider.durable_for_queue:
                    raise TriggerError(
                        TriggerErrorCode.PROVIDER_REFERENCE_EXPIRED,
                        "input.files",
                    )
            else:
                local = True
        if local:
            backend = self.client._artifact_store
            if not isinstance(
                backend, PgsqlArtifactStore | ObjectArtifactStore
            ):
                raise TriggerError(
                    TriggerErrorCode.ARTIFACT_NOT_DURABLE, "input.files"
                )
            backend._require_policy()
            if (
                isinstance(backend, ObjectArtifactStore)
                and backend._cipher is None
            ):
                raise TriggerError(
                    TriggerErrorCode.ARTIFACT_NOT_DURABLE, "input.encryption"
                )

    async def _stage(
        self, staging: ArtifactStagingOwner, values: tuple[ArtifactObject, ...]
    ) -> None:
        if isinstance(self.ownership, PgsqlArtifactOwnership):
            async with self.ownership.transaction() as unit:
                for value in sorted(values, key=lambda item: item.object_id):
                    await self.ownership.stage(
                        value, staging, unit_of_work=unit
                    )
        else:
            async with self.ownership.transaction() as memory:
                for value in values:
                    memory.stage(value, staging)

    async def prepare_registration(
        self,
        configuration: TriggerConfiguration,
        definition: TaskDefinition,
    ) -> PreparedTriggerRegistration:
        """Validate and encrypt static input before obtaining trigger locks."""
        assert isinstance(configuration, TriggerConfiguration)
        await self.admission.preflight()
        client = self.client
        if client._execution_deployment_id is None:
            raise TriggerError(
                TriggerErrorCode.DEPLOYMENT_MISMATCH, "deployment"
            )
        if (
            not client._raw_storage_allowed
            or client._encryption_provider is None
            or client._sanitizer(definition).policy.raw_retention_days <= 0
        ):
            raise TriggerError(
                TriggerErrorCode.CAPABILITY_UNAVAILABLE, "input.encryption"
            )
        identity = str(uuid4())
        staging = ArtifactStagingOwner(
            staging_id=identity,
            owner_scope=self.owner.value,
            recovery_id=identity,
        )
        bound = plain_value(
            bind_values(
                configuration.input,
                owner=self.owner,
                trigger_id=identity,
                revision=1,
                scheduled_at=client._clock(),
            )
        )
        # Binding a file descriptor would turn a static artifact into a
        # dynamic external input. File identity remains immutable per revision.
        if task_file_descriptors_from_input(
            definition, bound
        ) != task_file_descriptors_from_input(
            definition, configuration.input.value
        ):
            raise TriggerError(TriggerErrorCode.INVALID_BINDING, "input.files")
        self._durable_input(definition, bound)
        staged_backend = (
            StagedArtifactStore(
                client._artifact_store, self.ownership, staging
            )
            if isinstance(
                client._artifact_store,
                PgsqlArtifactStore | ObjectArtifactStore,
            )
            else None
        )
        try:
            task_input = await prepare_task_input(
                client, definition, input_value=bound, staging=staged_backend
            )
        except CancelledError as error:
            raise TriggerPreparationCancelledError(
                TriggerPreparationFailure(
                    staging=staging,
                    objects=(
                        tuple(staged_backend.objects) if staged_backend else ()
                    ),
                )
            ) from error
        except Exception as error:
            if staged_backend is not None and staged_backend.objects:
                raise TriggerPreparationFailure(
                    error=error,
                    staging=staging,
                    objects=tuple(staged_backend.objects),
                ) from error
            raise
        try:
            semantic_hash = configuration_hash(
                configuration,
                task_definition_id=task_input.definition_id,
                execution_deployment_id=client._execution_deployment_id,
            )
            encrypted = seal_input(
                RegisteredTriggerInput(
                    input=configuration.input,
                    files=task_input.materialized_files,
                ),
                client._encryption_provider,
                owner=self.owner,
                name=configuration.name,
                semantic_hash=semantic_hash,
            )
            registration = TriggerRegistration(
                name=configuration.name,
                task_definition_id=task_input.definition_id,
                execution_deployment_id=client._execution_deployment_id,
                semantic_hash=semantic_hash,
                schedule=configuration.schedule,
                input=encrypted,
                policy=configuration.policy,
                desired_enabled=configuration.desired_enabled,
                schedule_provenance=(
                    schedule_provenance(configuration.schedule.timezone)
                    if isinstance(configuration.schedule, CronTrigger)
                    else ScheduleProvenance(
                        semantics_version=1,
                        parser_version="builtin",
                        timezone_data="UTC",
                    )
                ),
            )
            prepared = PreparedTriggerRegistration(
                _authority=_REGISTRATION_AUTHORITY,
                registration=registration,
                task_input=task_input,
                staging=staging,
                objects=tuple(
                    ArtifactObject.from_ref(file.ref)
                    for file in task_input.materialized_files
                ),
            )
        except CancelledError as error:
            raise TriggerPreparationCancelledError(
                TriggerPreparationFailure(
                    staging=staging,
                    objects=(
                        tuple(staged_backend.objects) if staged_backend else ()
                    ),
                )
            ) from error
        except Exception as error:
            raise TriggerPreparationFailure(
                error=error,
                staging=staging,
                objects=(
                    tuple(staged_backend.objects) if staged_backend else ()
                ),
            ) from error
        try:
            await self._stage(staging, prepared.objects)
        except CancelledError as error:
            raise TriggerPreparationCancelledError(
                TriggerPreparationFailure(registration=prepared)
            ) from error
        except Exception as error:
            raise TriggerPreparationFailure(
                error=error, registration=prepared
            ) from error
        return prepared

    async def release_unused_admission(
        self, prepared: PreparedTriggerAdmission
    ) -> TriggerAdmissionResult:
        """Reconcile the decision fence before releasing unused staging."""
        for submission in prepared.submissions:
            self.client._assert_prepared_submission(submission)
        result = replace(
            await self.admission.recover(prepared.plan), prepared=prepared
        )
        if result.outcome == TriggerCommitOutcome.UNKNOWN:
            return result
        outcome = (
            TaskSubmissionOutcome.COMMITTED
            if result.outcome == TriggerCommitOutcome.COMMITTED
            else TaskSubmissionOutcome.NOT_COMMITTED
        )
        owners = tuple(
            ArtifactStagingOwner(
                staging_id=value.submission_id,
                owner_scope=value.owner_scope,
                recovery_id=value.submission_id,
            )
            for value in prepared.submissions
        )
        try:
            if isinstance(self.ownership, PgsqlArtifactOwnership):
                async with self.ownership.transaction() as unit:
                    for owner in owners:
                        await self.ownership.release_staging(
                            owner, outcome=outcome, unit_of_work=unit
                        )
            else:
                async with self.ownership.transaction() as memory:
                    for owner in owners:
                        memory.release_staging(owner, outcome=outcome)
        except CancelledError as error:
            raise TriggerAdmissionCancelledError(
                replace(result, cleanup_pending=True)
            ) from error
        except Exception:
            return replace(result, cleanup_pending=True)
        return result

    async def prepare_admission(
        self, plan: TriggerAdmissionPlan
    ) -> PreparedTriggerAdmission:
        """Revalidate occurrence inputs through TaskClient preparation."""
        await self.admission.preflight()
        definition = plan.snapshot.definition
        if (
            definition.owner_scope_id != self.owner
            or definition.execution_deployment_id
            != self.client._execution_deployment_id
        ):
            raise TriggerError(
                TriggerErrorCode.DEPLOYMENT_MISMATCH, "deployment"
            )
        decrypted = unseal_input(
            definition.input,
            self.decryption,
            owner=self.owner,
            name=definition.name,
            semantic_hash=definition.semantic_hash,
        )
        task = await self.client._store.get_definition(
            definition.task_definition_id
        )
        submissions: list[PreparedTaskSubmission] = []
        try:
            for decision in plan.decisions:
                if (
                    not isinstance(decision, PlannedOccurrence)
                    or plan.identity(decision) not in plan.admission_ids
                ):
                    continue
                value = plain_value(
                    bind_input(
                        decrypted.input, definition, decision.scheduled_at
                    )
                )
                self._durable_input(task.definition, value)
                restored = await restore_task_input(
                    self.client,
                    task.definition,
                    input_value=value,
                    materialized_files=decrypted.files,
                )
                if restored.definition_id != definition.task_definition_id:
                    raise TriggerError(
                        TriggerErrorCode.DEPLOYMENT_MISMATCH, "task"
                    )
                submission = await self.client.prepare_submission(
                    task.definition,
                    request=TaskSubmissionRequest(
                        input_value=value, available_at=decision.scheduled_at
                    ),
                    occurrence_id=plan.identity(decision),
                    prepared_input=restored,
                )
                submissions.append(submission)
                await self._stage(
                    ArtifactStagingOwner(
                        staging_id=submission.submission_id,
                        owner_scope=self.owner.value,
                        recovery_id=submission.submission_id,
                    ),
                    tuple(
                        ArtifactObject.from_ref(artifact.ref)
                        for artifact in submission.artifacts
                    ),
                )
        except CancelledError as error:
            raise TriggerPreparationCancelledError(
                TriggerPreparationFailure(submissions=tuple(submissions))
            ) from error
        except Exception as error:
            raise TriggerPreparationFailure(
                error=error, submissions=tuple(submissions)
            ) from error
        return PreparedTriggerAdmission(
            plan=plan, submissions=tuple(submissions)
        )
