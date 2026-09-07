"""Prepare reusable validated task input without allocating a task run."""

from .artifact_staging import StagedArtifactStore
from .definition import ObservabilitySinkType, RunMode, TaskDefinition
from .materialization import (
    TaskMaterializedFile,
    _task_file_descriptor_entries_from_input,
    materialize_task_input_files,
    task_file_descriptors_from_input,
)
from .schema import task_definition_schema_base_path
from .skills import (
    task_definition_with_skills_identity,
    task_skill_audit_event_publisher,
)
from .store import TaskSnapshotValue, freeze_snapshot_value

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .client import TaskClient


_INPUT_AUTHORITY = object()


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedTaskInput:
    """Retain validated materialization without inventing a task run.

    The client instance binds privacy, target and storage capabilities.
    Reconstructing an input after a restart requires validating it again.
    """

    definition: TaskDefinition
    definition_id: str
    input_value: TaskSnapshotValue = field(repr=False)
    materialized_files: tuple[TaskMaterializedFile, ...] = field(repr=False)
    _client: "TaskClient" = field(repr=False, compare=False)
    _authority: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        assert self._authority is _INPUT_AUTHORITY
        assert isinstance(self.definition, TaskDefinition)
        assert self.definition_id.strip()
        assert isinstance(self.materialized_files, tuple)
        assert all(
            isinstance(value, TaskMaterializedFile)
            for value in self.materialized_files
        )
        object.__setattr__(
            self, "input_value", freeze_snapshot_value(self.input_value)
        )

    def files_for_submission(
        self, client: "TaskClient", definition_id: str, input_value: object
    ) -> tuple[TaskMaterializedFile, ...]:
        """Clone per-run reference IDs after checking unchanged descriptors."""
        assert client is self._client
        assert definition_id == self.definition_id
        assert task_file_descriptors_from_input(
            self.definition, input_value
        ) == (
            task_file_descriptors_from_input(self.definition, self.input_value)
        )
        return tuple(
            replace(value, ref=replace(value.ref, artifact_id=str(uuid4())))
            for value in self.materialized_files
        )


async def resolve_submission_definition(
    client: "TaskClient", definition: TaskDefinition
) -> TaskDefinition:
    """Resolve schemas and skills through the configured client policies."""
    schema_base_path = task_definition_schema_base_path(definition)
    definition = await client._resolve_definition_schemas(definition)
    sanitizer = client._sanitizer(definition)
    return await task_definition_with_skills_identity(
        definition,
        event_manager=task_skill_audit_event_publisher(
            sanitizer=sanitizer,
            event_observer=client._event_observer,
            metrics_event_observer=(
                client._metrics_event_observer
                if definition.observability.metrics
                else None
            ),
            trace_event_observer=(
                client._trace_event_observer
                if definition.observability.trace
                else None
            ),
            observability_sink=(
                client._observability_sink
                if definition.observability.sinks
                != (ObservabilitySinkType.NOOP,)
                else None
            ),
        ),
        schema_base_path=schema_base_path,
    )


async def prepare_task_input(
    client: "TaskClient",
    definition: TaskDefinition,
    *,
    input_value: object,
    staging: StagedArtifactStore | None = None,
) -> PreparedTaskInput:
    """Validate task capability and materialize files outside admission locks.

    Perform the read-only queue/schema check before definition registration
    or external writes. This is the reusable part of submission, not queue
    admission and not a deployment closure attestation.
    """
    snapshot = freeze_snapshot_value(input_value)
    assert definition.run.mode == RunMode.QUEUE
    assert client._queue is not None
    client._queue.validate_submission_store(client._store)
    await client._queue.preflight_submission()
    definition = await resolve_submission_definition(client, definition)
    validation = await client.validate(definition, input_value=snapshot)
    validation.raise_for_issues()
    definition_id = await client._definition_hash_value(definition)
    await client._store.register_definition(
        definition, definition_hash=definition_id
    )
    if staging is not None:
        assert staging.backend is client._artifact_store
    files = await materialize_task_input_files(
        definition,
        snapshot,
        roots=client._input_roots,
        artifact_store=staging or client._artifact_store,
        hmac_provider=client._hmac_provider,
        remote_url_policy=client._remote_url_policy,
        remote_url_http_client=client._remote_url_http_client,
        remote_url_resolver=client._remote_url_resolver,
    )
    return PreparedTaskInput(
        definition=definition,
        definition_id=definition_id,
        input_value=snapshot,
        materialized_files=files,
        _client=client,
        _authority=_INPUT_AUTHORITY,
    )


async def restore_task_input(
    client: "TaskClient",
    definition: TaskDefinition,
    *,
    input_value: object,
    materialized_files: tuple[TaskMaterializedFile, ...],
) -> PreparedTaskInput:
    """Revalidate registered files without rereading local sources."""
    snapshot = freeze_snapshot_value(input_value)
    assert definition.run.mode == RunMode.QUEUE
    assert client._queue is not None
    client._queue.validate_submission_store(client._store)
    await client._queue.preflight_submission()
    definition = await resolve_submission_definition(client, definition)
    validation = await client.validate(definition, input_value=snapshot)
    validation.raise_for_issues()
    definition_id = await client._definition_hash_value(definition)
    entries, issues = _task_file_descriptor_entries_from_input(
        definition, snapshot
    )
    assert not issues
    expected = tuple(
        entry
        for entry in entries
        if entry.descriptor.source_kind.value in {"local_path", "remote_url"}
    )
    assert len(expected) == len(materialized_files)
    if materialized_files:
        assert client._artifact_store is not None
        for entry, file in zip(expected, materialized_files, strict=True):
            assert entry.path == file.descriptor_path
            assert entry.descriptor == file.descriptor
            stat = await client._artifact_store.stat(file.ref)
            assert stat.sha256 == file.ref.sha256
            assert stat.size_bytes == file.ref.size_bytes
    return PreparedTaskInput(
        definition=definition,
        definition_id=definition_id,
        input_value=snapshot,
        materialized_files=materialized_files,
        _client=client,
        _authority=_INPUT_AUTHORITY,
    )
