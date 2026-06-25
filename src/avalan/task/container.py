from ..container import (
    ContainerBackend,
    ContainerDurablePlanKind,
    ContainerExecutionScope,
    ContainerMountAccess,
    ContainerMountDeclaration,
    ContainerMountType,
    ContainerNormalizedRunPlan,
    ContainerNormalizedRuntimeEnvelopePlan,
    ContainerOutputArtifact,
    ContainerOutputContract,
    ContainerOutputContractType,
    ContainerPlanRequest,
    ContainerPlanRequestKind,
    ContainerRunPlan,
    ContainerRuntimeEnvelopeKind,
    normalize_container_run_plan,
    normalize_runtime_envelope_plan,
    output_contracts_from_policy,
)
from .artifact import (
    ArtifactStore,
    TaskArtifactProvenance,
    TaskOutputArtifact,
)
from .context import TaskInputFile
from .definition import TaskDefinition, TaskOutputType
from .store import (
    TaskAttempt,
    TaskRun,
    TaskSnapshotMetadata,
    freeze_snapshot_metadata,
)

from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import cast

TASK_CONTAINER_METADATA_KEY = "container"
TASK_CONTAINER_ATTEMPT_KEY = "attempt"
TASK_CONTAINER_ATTEMPT_LIFECYCLE_KEY = "attempt_lifecycle"
TASK_CONTAINER_WORKER_ENVELOPE_KEY = "worker_envelope"
TASK_CONTAINER_INPUT_MOUNTS_KEY = "input_mounts"
TASK_CONTAINER_ISOLATION_KEY = "isolation"
TASK_CONTAINER_CANONICAL_ATTEMPT_ID = "canonical"
TASK_CONTAINER_CANONICAL_REQUEST_ID = "canonical"
TASK_CONTAINER_ISOLATION_MODE = "container"
TASK_CONTAINER_ISOLATION_PLAN_FIELDS: frozenset[str] = frozenset(
    {
        "backend",
        "effective_mode",
        "elevation_rung",
        "mode",
        "plan_fingerprint",
        "plan_kind",
        "policy_version",
        "profile_name",
        "profile_registry_id",
        "request_kind",
        "requested_mode",
        "scope",
    }
)


class TaskContainerVerificationError(RuntimeError):
    code: str
    path: str
    hint: str

    def __init__(
        self,
        *,
        code: str,
        path: str,
        message: str,
        hint: str,
    ) -> None:
        self.code = code
        self.path = path
        self.hint = hint
        super().__init__(message)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskContainerPlans:
    attempt: ContainerNormalizedRunPlan | None = None
    worker_envelope: ContainerNormalizedRuntimeEnvelopePlan | None = None

    @property
    def enabled(self) -> bool:
        return self.attempt is not None or self.worker_envelope is not None

    def to_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {}
        if self.attempt is not None:
            metadata[TASK_CONTAINER_ATTEMPT_KEY] = (
                self.attempt.to_metadata().to_dict()
            )
        if self.worker_envelope is not None:
            metadata[TASK_CONTAINER_WORKER_ENVELOPE_KEY] = (
                self.worker_envelope.to_metadata().to_dict()
            )
        isolation = _isolation_metadata(
            attempt=self.attempt,
            attempt_lifecycle=None,
            worker_envelope=self.worker_envelope,
        )
        if isolation:
            metadata[TASK_CONTAINER_ISOLATION_KEY] = isolation
        return metadata


def task_container_canonical_value(
    definition: TaskDefinition,
) -> dict[str, object]:
    assert isinstance(definition, TaskDefinition)
    container = definition.container
    value = container.to_dict()
    attempt = _attempt_plan(
        definition,
        request_id=TASK_CONTAINER_CANONICAL_REQUEST_ID,
        attempt_id=TASK_CONTAINER_CANONICAL_ATTEMPT_ID,
    )
    worker_envelope = _worker_envelope_plan(
        definition,
        request_id=TASK_CONTAINER_CANONICAL_REQUEST_ID,
    )
    value["attempt_spec"] = _plan_spec(attempt)
    value["worker_envelope_spec"] = _plan_spec(worker_envelope)
    value["isolation_spec"] = _isolation_metadata(
        attempt=attempt,
        attempt_lifecycle=None,
        worker_envelope=worker_envelope,
    )
    return value


def task_container_request_metadata(
    definition: TaskDefinition,
    *,
    input_mounts: tuple[dict[str, object], ...] = (),
) -> TaskSnapshotMetadata:
    assert isinstance(definition, TaskDefinition)
    assert isinstance(input_mounts, tuple)
    metadata: dict[str, object] = {}
    attempt = _attempt_plan(
        definition,
        request_id=TASK_CONTAINER_CANONICAL_REQUEST_ID,
        attempt_id=TASK_CONTAINER_CANONICAL_ATTEMPT_ID,
    )
    worker_envelope = _worker_envelope_plan(
        definition,
        request_id=TASK_CONTAINER_CANONICAL_REQUEST_ID,
    )
    if attempt is not None:
        metadata[TASK_CONTAINER_ATTEMPT_KEY] = _plan_spec(attempt)
        if input_mounts:
            metadata[TASK_CONTAINER_ATTEMPT_LIFECYCLE_KEY] = (
                _attempt_lifecycle_plan_spec(attempt, input_mounts)
            )
    if worker_envelope is not None:
        metadata[TASK_CONTAINER_WORKER_ENVELOPE_KEY] = _plan_spec(
            worker_envelope
        )
    isolation = _isolation_metadata(
        attempt=attempt,
        attempt_lifecycle=(
            _attempt_lifecycle_plan(attempt, input_mounts)
            if attempt is not None and input_mounts
            else None
        ),
        worker_envelope=worker_envelope,
    )
    if isolation:
        metadata[TASK_CONTAINER_ISOLATION_KEY] = isolation
    return freeze_snapshot_metadata(metadata)


def task_container_run_metadata(
    definition: TaskDefinition,
    metadata: Mapping[str, object] | None,
    *,
    input_mounts: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    assert isinstance(definition, TaskDefinition)
    merged = dict(metadata or {})
    container_metadata = task_container_request_metadata(
        definition,
        input_mounts=input_mounts,
    )
    if container_metadata:
        merged[TASK_CONTAINER_METADATA_KEY] = container_metadata
    return merged


def task_container_user_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    visible = dict(metadata)
    visible.pop(TASK_CONTAINER_METADATA_KEY, None)
    return visible


def task_container_plans(
    definition: TaskDefinition,
    *,
    run: TaskRun,
    attempt: TaskAttempt,
) -> TaskContainerPlans:
    assert isinstance(definition, TaskDefinition)
    assert isinstance(run, TaskRun)
    assert isinstance(attempt, TaskAttempt)
    return TaskContainerPlans(
        attempt=_attempt_plan(
            definition,
            request_id=run.run_id,
            attempt_id=attempt.attempt_id,
        ),
        worker_envelope=_worker_envelope_plan(
            definition,
            request_id=run.run_id,
        ),
    )


def verify_task_container_request(
    definition: TaskDefinition,
    *,
    run: TaskRun,
    attempt: TaskAttempt,
    input_mounts: tuple[dict[str, object], ...] = (),
    allow_dynamic_input_mounts: bool = False,
) -> TaskContainerPlans:
    assert isinstance(input_mounts, tuple)
    assert isinstance(allow_dynamic_input_mounts, bool)
    plans = task_container_plans(definition, run=run, attempt=attempt)
    if not plans.enabled:
        return plans
    raw = run.request.metadata.get(TASK_CONTAINER_METADATA_KEY)
    if not isinstance(raw, Mapping):
        raise _verification_error(
            "container.plan_missing",
            TASK_CONTAINER_METADATA_KEY,
            "Queued task container policy metadata is missing.",
            "Enqueue the task through a trusted task client.",
        )
    _assert_metadata_matches_plan(
        raw,
        TASK_CONTAINER_ATTEMPT_KEY,
        plans.attempt,
    )
    _assert_metadata_matches_plan(
        raw,
        TASK_CONTAINER_WORKER_ENVELOPE_KEY,
        plans.worker_envelope,
    )
    attempt_lifecycle = (
        _attempt_lifecycle_plan(plans.attempt, input_mounts)
        if input_mounts
        else None
    )
    validate_attempt_lifecycle = not (
        allow_dynamic_input_mounts and input_mounts
    )
    if validate_attempt_lifecycle:
        _assert_metadata_matches_plan(
            raw,
            TASK_CONTAINER_ATTEMPT_LIFECYCLE_KEY,
            attempt_lifecycle,
        )
    _assert_isolation_metadata_matches_plans(
        raw,
        attempt=plans.attempt,
        attempt_lifecycle=attempt_lifecycle,
        worker_envelope=plans.worker_envelope,
        validate_attempt_lifecycle=validate_attempt_lifecycle,
    )
    return plans


def task_container_input_mount_manifest(
    files: tuple[TaskInputFile, ...],
    *,
    allowed_roots: tuple[str | Path, ...] = (),
) -> tuple[dict[str, object], ...]:
    assert isinstance(files, tuple)
    roots = tuple(Path(root).resolve(strict=False) for root in allowed_roots)
    mounts: list[dict[str, object]] = []
    for index, file in enumerate(files):
        assert isinstance(file, TaskInputFile)
        source_kind = _input_file_source_kind(file)
        if source_kind is None:
            continue
        mount: dict[str, object] = {
            "mount_type": "input",
            "source_kind": source_kind,
            "target": f"/inputs/{index}",
        }
        if file.artifact_ref is not None:
            mount["artifact_id"] = file.artifact_ref.artifact_id
            source = file.artifact_ref.metadata.get("container_mount_source")
            if isinstance(source, str) and _is_allowed_mount_source(
                source,
                roots,
            ):
                mount["source"] = source
        if file.media_type is not None:
            mount["media_type"] = file.media_type
        if file.size_bytes is not None:
            mount["size_bytes"] = file.size_bytes
        mounts.append(mount)
    return tuple(mounts)


def task_container_lifecycle_run_plan(
    plans: TaskContainerPlans,
    *,
    input_mounts: tuple[dict[str, object], ...] = (),
) -> ContainerRunPlan | None:
    assert isinstance(plans, TaskContainerPlans)
    assert isinstance(input_mounts, tuple)
    base = _base_run_plan(plans)
    if base is None:
        return None
    mounts = (*base.mounts, *_input_mount_declarations(input_mounts))
    return replace(base, mounts=mounts)


def task_container_unsupported_input_mount_path(
    input_mounts: tuple[dict[str, object], ...],
) -> str | None:
    assert isinstance(input_mounts, tuple)
    for index, mount in enumerate(input_mounts):
        if mount.get("source_kind") != "artifact":
            return f"container.input_mounts[{index}].source_kind"
        if not isinstance(mount.get("source"), str):
            return f"container.input_mounts[{index}].source"
    return None


def task_container_output_contract(
    definition: TaskDefinition,
    plans: TaskContainerPlans,
) -> ContainerOutputContract | None:
    assert isinstance(definition, TaskDefinition)
    assert isinstance(plans, TaskContainerPlans)
    if plans.attempt is None:
        return None
    settings = definition.container.attempt
    if settings is None or settings.profile is None:
        return None
    artifact_type = _container_output_artifact_type(definition.output.type)
    if artifact_type is None:
        return None
    for contract in output_contracts_from_policy(settings.profile.output):
        if contract.contract_type is artifact_type:
            return contract
    return None


async def task_container_output_artifacts(
    definition: TaskDefinition,
    artifacts: tuple[object, ...],
    *,
    run_id: str,
    attempt_id: str,
    artifact_store: ArtifactStore | None,
) -> object:
    assert isinstance(definition, TaskDefinition)
    if artifact_store is None:
        raise _verification_error(
            "container.output_unsupported",
            "container.output",
            "Task container output requires an artifact store.",
            "Configure a trusted artifact store for task container outputs.",
        )
    if definition.output.type is TaskOutputType.FILE and len(artifacts) != 1:
        raise _verification_error(
            "container.output_unsupported",
            "container.output",
            "Task file output requires exactly one container artifact.",
            "Return one accepted task artifact for file output contracts.",
        )
    _validate_container_output_artifact_limits(definition, artifacts)
    records = tuple(
        [
            await _container_output_artifact(
                artifact,
                run_id=run_id,
                attempt_id=attempt_id,
                artifact_store=artifact_store,
            )
            for artifact in artifacts
        ]
    )
    if definition.output.type is TaskOutputType.FILE:
        return records[0]
    return records


async def _container_output_artifact(
    artifact: object,
    *,
    run_id: str,
    attempt_id: str,
    artifact_store: ArtifactStore,
) -> TaskOutputArtifact:
    assert isinstance(artifact, ContainerOutputArtifact)
    digest = artifact.digest.removeprefix("sha256:")
    content = artifact.content
    if content is None:
        raise _verification_error(
            "container.output_unsupported",
            "container.output",
            "Task container output did not include copied artifact bytes.",
            "Use a backend that returns bounded copied output bytes.",
        )
    if sha256(content).hexdigest() != digest:
        raise _verification_error(
            "container.output_unsupported",
            "container.output.digest",
            "Task container output digest does not match copied bytes.",
            "Retry with a backend that verifies copied output artifacts.",
        )
    artifact_id = _container_output_artifact_id(
        run_id,
        attempt_id,
        artifact.path,
        digest,
    )
    metadata = {
        "container_artifact_type": (
            cast(
                ContainerOutputContractType,
                artifact.artifact_type,
            ).value
        ),
        "container_path": artifact.path,
        "quarantined": artifact.quarantined,
    }
    ref = await artifact_store.put(
        content,
        artifact_id=artifact_id,
        media_type=artifact.media_type,
        metadata=metadata,
    )
    return TaskOutputArtifact(
        ref=ref,
        provenance=TaskArtifactProvenance(
            source_run_id=run_id,
            source_attempt_id=attempt_id,
            operation="container_output",
            metadata={"container_path": artifact.path},
        ),
        metadata=metadata,
    )


def task_container_event_payload(
    *,
    status: str,
    plans: TaskContainerPlans,
    input_mounts: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    payload: dict[str, object] = {"status": status}
    if plans.attempt is not None:
        payload.update(_safe_plan_payload(plans.attempt))
    if plans.worker_envelope is not None:
        if plans.attempt is None:
            payload.update(_safe_plan_payload(plans.worker_envelope.run_plan))
        payload["worker_envelope_plan_fingerprint"] = (
            plans.worker_envelope.plan_fingerprint
        )
    if input_mounts:
        payload["mount_count"] = len(input_mounts)
        payload["artifact_count"] = sum(
            1
            for mount in input_mounts
            if mount.get("source_kind") == "artifact"
        )
    return payload


def _attempt_plan(
    definition: TaskDefinition,
    *,
    request_id: str,
    attempt_id: str,
) -> ContainerNormalizedRunPlan | None:
    settings = definition.container.attempt
    if settings is None or not settings.enabled:
        return None
    return normalize_container_run_plan(
        settings,
        _plan_request(
            definition,
            request_kind=ContainerPlanRequestKind.TASK_ATTEMPT,
            request_id=request_id,
            attempt_id=attempt_id,
            scope=settings.scope,
        ),
    )


def _worker_envelope_plan(
    definition: TaskDefinition,
    *,
    request_id: str,
) -> ContainerNormalizedRuntimeEnvelopePlan | None:
    settings = definition.container.worker_envelope
    if settings is None or not settings.enabled:
        return None
    return normalize_runtime_envelope_plan(
        settings,
        _plan_request(
            definition,
            request_kind=ContainerPlanRequestKind.RUNTIME_ENVELOPE,
            request_id=request_id,
            attempt_id=None,
            scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
        ),
        envelope_kind=ContainerRuntimeEnvelopeKind.TASK_WORKER,
        readiness_timeout_seconds=(
            definition.container.readiness_timeout_seconds
        ),
    )


def _plan_request(
    definition: TaskDefinition,
    *,
    request_kind: ContainerPlanRequestKind,
    request_id: str,
    attempt_id: str | None,
    scope: ContainerExecutionScope | str,
) -> ContainerPlanRequest:
    target_type = definition.execution.type.value
    target_ref = definition.execution.ref
    return ContainerPlanRequest(
        request_kind=request_kind,
        logical_name=definition.task.name,
        command="avalan-task",
        argv=("avalan-task", target_type, target_ref),
        cwd="/workspace",
        scope=scope,
        request_id=request_id,
        attempt_id=attempt_id,
    )


def _plan_spec(
    plan: (
        ContainerNormalizedRunPlan
        | ContainerNormalizedRuntimeEnvelopePlan
        | None
    ),
) -> dict[str, object] | None:
    if plan is None:
        return None
    if isinstance(plan, ContainerNormalizedRuntimeEnvelopePlan):
        run_plan = plan.run_plan
        plan_fingerprint = plan.plan_fingerprint
    else:
        run_plan = plan
        plan_fingerprint = plan.plan_fingerprint
    return {
        "profile_name": run_plan.run_plan.profile_name,
        "profile_registry_id": run_plan.profile_registry_id,
        "policy_version": run_plan.run_plan.policy_version,
        "plan_fingerprint": plan_fingerprint,
    }


def _attempt_lifecycle_plan(
    plan: ContainerNormalizedRunPlan | None,
    input_mounts: tuple[dict[str, object], ...],
) -> ContainerNormalizedRunPlan | None:
    if plan is None:
        return None
    run_plan = task_container_lifecycle_run_plan(
        TaskContainerPlans(attempt=plan),
        input_mounts=input_mounts,
    )
    assert run_plan is not None
    return replace(plan, run_plan=run_plan)


def _attempt_lifecycle_plan_spec(
    plan: ContainerNormalizedRunPlan,
    input_mounts: tuple[dict[str, object], ...],
) -> dict[str, object] | None:
    return _plan_spec(_attempt_lifecycle_plan(plan, input_mounts))


def _isolation_metadata(
    *,
    attempt: ContainerNormalizedRunPlan | None,
    attempt_lifecycle: ContainerNormalizedRunPlan | None,
    worker_envelope: ContainerNormalizedRuntimeEnvelopePlan | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if attempt is not None:
        metadata[TASK_CONTAINER_ATTEMPT_KEY] = _isolation_plan_spec(attempt)
    if attempt_lifecycle is not None:
        metadata[TASK_CONTAINER_ATTEMPT_LIFECYCLE_KEY] = _isolation_plan_spec(
            attempt_lifecycle
        )
    if worker_envelope is not None:
        metadata[TASK_CONTAINER_WORKER_ENVELOPE_KEY] = _isolation_plan_spec(
            worker_envelope
        )
    return metadata


def _isolation_plan_spec(
    plan: ContainerNormalizedRunPlan | ContainerNormalizedRuntimeEnvelopePlan,
) -> dict[str, object]:
    if isinstance(plan, ContainerNormalizedRuntimeEnvelopePlan):
        run_plan = plan.run_plan
        plan_fingerprint = plan.plan_fingerprint
        plan_kind = ContainerDurablePlanKind.RUNTIME_ENVELOPE
        scope = cast(ContainerExecutionScope, plan.envelope_plan.scope)
    else:
        run_plan = plan
        plan_fingerprint = plan.plan_fingerprint
        plan_kind = ContainerDurablePlanKind.RUN
        scope = cast(ContainerExecutionScope, run_plan.request.scope)
    backend = cast(ContainerBackend, run_plan.run_plan.backend)
    request_kind = cast(
        ContainerPlanRequestKind, run_plan.request.request_kind
    )
    return {
        "mode": TASK_CONTAINER_ISOLATION_MODE,
        "backend": backend.value,
        "profile_name": run_plan.run_plan.profile_name,
        "profile_registry_id": run_plan.profile_registry_id,
        "policy_version": run_plan.run_plan.policy_version,
        "scope": scope.value,
        "plan_fingerprint": plan_fingerprint,
        "requested_mode": TASK_CONTAINER_ISOLATION_MODE,
        "effective_mode": TASK_CONTAINER_ISOLATION_MODE,
        "elevation_rung": TASK_CONTAINER_ISOLATION_MODE,
        "plan_kind": plan_kind.value,
        "request_kind": request_kind.value,
    }


def _assert_metadata_matches_plan(
    raw: Mapping[str, object],
    key: str,
    plan: (
        ContainerNormalizedRunPlan
        | ContainerNormalizedRuntimeEnvelopePlan
        | None
    ),
) -> None:
    expected = _plan_spec(plan)
    actual = raw.get(key)
    if expected is None:
        if actual is not None:
            raise _verification_error(
                "container.plan_unexpected",
                f"{TASK_CONTAINER_METADATA_KEY}.{key}",
                "Queued task container metadata includes an unexpected plan.",
                "Remove untrusted container metadata from the task request.",
            )
        return
    if not isinstance(actual, Mapping):
        raise _verification_error(
            "container.plan_missing",
            f"{TASK_CONTAINER_METADATA_KEY}.{key}",
            "Queued task container plan metadata is missing.",
            "Enqueue the task through a trusted task client.",
        )
    for field_name, expected_value in expected.items():
        if actual.get(field_name) != expected_value:
            raise _verification_error(
                "container.plan_mismatch",
                f"{TASK_CONTAINER_METADATA_KEY}.{key}.{field_name}",
                "Queued task container policy metadata is stale.",
                "Re-enqueue the task with the current trusted container "
                "policy.",
            )


def _assert_isolation_metadata_matches_plan(
    raw_isolation: Mapping[str, object],
    key: str,
    plan: ContainerNormalizedRunPlan | ContainerNormalizedRuntimeEnvelopePlan,
) -> None:
    expected = _isolation_plan_spec(plan)
    actual = raw_isolation.get(key)
    path = (
        f"{TASK_CONTAINER_METADATA_KEY}.{TASK_CONTAINER_ISOLATION_KEY}.{key}"
    )
    if not isinstance(actual, Mapping):
        raise _verification_error(
            "container.plan_missing",
            path,
            "Queued task isolation plan metadata is missing.",
            "Enqueue the task through a trusted isolation-aware task client.",
        )
    for field_name, expected_value in expected.items():
        if actual.get(field_name) != expected_value:
            raise _verification_error(
                "container.plan_mismatch",
                f"{path}.{field_name}",
                "Queued task isolation metadata is stale.",
                "Re-enqueue the task with the current trusted isolation "
                "policy.",
            )
    extra = set(actual) - set(expected)
    if extra:
        field_name = sorted(extra, key=str)[0]
        raise _verification_error(
            "container.plan_unexpected",
            f"{path}.{field_name}",
            "Queued task isolation metadata includes unsupported authority.",
            "Remove untrusted isolation metadata from the task request.",
        )


def _assert_isolation_metadata_matches_plans(
    raw: Mapping[str, object],
    *,
    attempt: ContainerNormalizedRunPlan | None,
    attempt_lifecycle: ContainerNormalizedRunPlan | None,
    worker_envelope: ContainerNormalizedRuntimeEnvelopePlan | None,
    validate_attempt_lifecycle: bool,
) -> None:
    raw_isolation = raw.get(TASK_CONTAINER_ISOLATION_KEY)
    if raw_isolation is None:
        return
    if not isinstance(raw_isolation, Mapping):
        raise _verification_error(
            "container.plan_missing",
            f"{TASK_CONTAINER_METADATA_KEY}.{TASK_CONTAINER_ISOLATION_KEY}",
            "Queued task isolation metadata is missing.",
            "Enqueue the task through a trusted isolation-aware task client.",
        )

    expected: dict[
        str,
        ContainerNormalizedRunPlan | ContainerNormalizedRuntimeEnvelopePlan,
    ] = {}
    if attempt is not None:
        expected[TASK_CONTAINER_ATTEMPT_KEY] = attempt
    if worker_envelope is not None:
        expected[TASK_CONTAINER_WORKER_ENVELOPE_KEY] = worker_envelope
    if validate_attempt_lifecycle and attempt_lifecycle is not None:
        expected[TASK_CONTAINER_ATTEMPT_LIFECYCLE_KEY] = attempt_lifecycle

    allowed_keys = set(expected)
    if not validate_attempt_lifecycle and attempt_lifecycle is not None:
        allowed_keys.add(TASK_CONTAINER_ATTEMPT_LIFECYCLE_KEY)
    extra = set(raw_isolation) - allowed_keys
    if extra:
        key = sorted(extra, key=str)[0]
        raise _verification_error(
            "container.plan_unexpected",
            f"{TASK_CONTAINER_METADATA_KEY}."
            f"{TASK_CONTAINER_ISOLATION_KEY}.{key}",
            "Queued task isolation metadata includes an unexpected plan.",
            "Remove untrusted isolation metadata from the task request.",
        )

    for key, plan in expected.items():
        _assert_isolation_metadata_matches_plan(raw_isolation, key, plan)
    if (
        not validate_attempt_lifecycle
        and attempt_lifecycle is not None
        and raw_isolation.get(TASK_CONTAINER_ATTEMPT_LIFECYCLE_KEY) is not None
    ):
        _assert_isolation_plan_metadata_supported(
            raw_isolation,
            TASK_CONTAINER_ATTEMPT_LIFECYCLE_KEY,
        )


def _assert_isolation_plan_metadata_supported(
    raw_isolation: Mapping[str, object],
    key: str,
) -> None:
    path = (
        f"{TASK_CONTAINER_METADATA_KEY}.{TASK_CONTAINER_ISOLATION_KEY}.{key}"
    )
    actual = raw_isolation.get(key)
    if not isinstance(actual, Mapping):
        raise _verification_error(
            "container.plan_missing",
            path,
            "Queued task isolation plan metadata is missing.",
            "Enqueue the task through a trusted isolation-aware task client.",
        )
    extra = set(actual) - TASK_CONTAINER_ISOLATION_PLAN_FIELDS
    if extra:
        field_name = sorted(extra, key=str)[0]
        raise _verification_error(
            "container.plan_unexpected",
            f"{path}.{field_name}",
            "Queued task isolation metadata includes unsupported authority.",
            "Remove untrusted isolation metadata from the task request.",
        )


def _safe_plan_payload(plan: ContainerNormalizedRunPlan) -> dict[str, object]:
    backend = cast(ContainerBackend, plan.run_plan.backend)
    request_kind = cast(ContainerPlanRequestKind, plan.request.request_kind)
    return {
        "backend": backend.value,
        "profile_name": plan.run_plan.profile_name,
        "profile_registry_id": plan.profile_registry_id,
        "policy_version": plan.run_plan.policy_version,
        "plan_fingerprint": plan.plan_fingerprint,
        "request_kind": request_kind.value,
    }


def _input_file_source_kind(file: TaskInputFile) -> str | None:
    if file.artifact_ref is not None:
        return "artifact"
    if file.provider_reference is not None:
        return "provider"
    if file.logical_path.startswith("inline:"):
        return "inline"
    return None


def _base_run_plan(plans: TaskContainerPlans) -> ContainerRunPlan | None:
    if plans.attempt is not None:
        return plans.attempt.run_plan
    if plans.worker_envelope is not None:
        return plans.worker_envelope.run_plan.run_plan
    return None


def _input_mount_declarations(
    input_mounts: tuple[dict[str, object], ...],
) -> tuple[ContainerMountDeclaration, ...]:
    declarations: list[ContainerMountDeclaration] = []
    for mount in input_mounts:
        source = mount.get("source")
        target = mount.get("target")
        source_kind = mount.get("source_kind")
        if source_kind != "artifact":
            continue
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        declarations.append(
            ContainerMountDeclaration(
                source=source,
                target=target,
                mount_type=ContainerMountType.INPUT,
                access=ContainerMountAccess.READ,
            )
        )
    return tuple(declarations)


def _is_allowed_mount_source(source: str, roots: tuple[Path, ...]) -> bool:
    if not roots:
        return False
    try:
        path = Path(source).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False
    return any(_is_relative_to(path, root) for root in roots)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_container_output_artifact_limits(
    definition: TaskDefinition,
    artifacts: tuple[object, ...],
) -> None:
    count_limits = (
        definition.artifact.max_count,
        definition.limits.artifact_count,
    )
    for limit in count_limits:
        if limit is not None and len(artifacts) > limit:
            raise _verification_error(
                "container.output_unsupported",
                "container.output",
                "Task container output exceeds artifact count limits.",
                "Return fewer accepted container output artifacts.",
            )
    for artifact in artifacts:
        assert isinstance(artifact, ContainerOutputArtifact)
        byte_limits = (
            definition.artifact.max_bytes,
            definition.limits.artifact_bytes,
        )
        for limit in byte_limits:
            if limit is not None and artifact.size_bytes > limit:
                raise _verification_error(
                    "container.output_unsupported",
                    "container.output",
                    "Task container output exceeds artifact byte limits.",
                    "Return smaller accepted container output artifacts.",
                )


def _container_output_artifact_type(
    output_type: TaskOutputType,
) -> ContainerOutputContractType | None:
    if output_type in {
        TaskOutputType.FILE,
        TaskOutputType.FILE_ARRAY,
        TaskOutputType.ARTIFACT_ARRAY,
    }:
        return ContainerOutputContractType.TASK_ARTIFACT
    return None


def _container_output_artifact_id(
    run_id: str,
    attempt_id: str,
    path: str,
    digest: str,
) -> str:
    value = sha256(
        f"{run_id}:{attempt_id}:{path}:{digest}".encode("utf-8")
    ).hexdigest()
    return f"container-output-{value[:32]}"


def _verification_error(
    code: str,
    path: str,
    message: str,
    hint: str,
) -> TaskContainerVerificationError:
    return TaskContainerVerificationError(
        code=code,
        path=path,
        message=message,
        hint=hint,
    )
