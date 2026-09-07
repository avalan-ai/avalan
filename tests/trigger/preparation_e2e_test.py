"""Exercise validated registration, restart restoration and shared bytes."""

from .records_test import NOW, OWNER

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta
from json import dumps
from os import urandom
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from task.artifacts.object_store_test import FakeObjectClient
from task_deployment_helpers import configure_fixture_deployment

from avalan.task.artifacts.object_store import (
    ObjectArtifactEncryption,
    ObjectArtifactStore,
    ObjectArtifactStorePolicy,
)
from avalan.task.client import TaskClient
from avalan.task.context import TaskTargetContext
from avalan.task.definition import (
    IdempotencyMode,
    PrivacyAction,
    TaskArtifactPolicy,
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskPrivacyPolicy,
    TaskRunPolicy,
)
from avalan.task.privacy import EncryptedPrivacyValue, TaskKeyPurpose
from avalan.task.queues.memory_submission import (
    MemoryTaskSubmissionParticipant,
)
from avalan.task.retention import TaskRetentionService
from avalan.task.state import TaskRunState
from avalan.task.stores.memory import InMemoryTaskStore
from avalan.trigger.admission import TriggerCommitOutcome
from avalan.trigger.apply import TriggerRegistrationService
from avalan.trigger.definition import (
    IntervalTrigger,
    TriggerConfiguration,
    TriggerInput,
)
from avalan.trigger.error import TriggerErrorCode
from avalan.trigger.plan import plan_admission
from avalan.trigger.preparation import TriggerPreparationService
from avalan.trigger.stores.memory import InMemoryTriggerStore
from avalan.trigger.stores.memory_admission import MemoryTriggerAdmissionStore


class ContextCipher:
    """Authenticate both fixture payload and object cipher contexts."""

    def _encrypt(
        self, value: bytes, context: Mapping[str, str] | None
    ) -> bytes:
        nonce = urandom(12)
        return nonce + AESGCM(b"k" * 32).encrypt(
            nonce, value, dumps(dict(context or {}), sort_keys=True).encode()
        )

    def _decrypt(
        self, value: bytes, context: Mapping[str, str] | None
    ) -> bytes:
        return AESGCM(b"k" * 32).decrypt(
            value[:12],
            value[12:],
            dumps(dict(context or {}), sort_keys=True).encode(),
        )

    def encrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> EncryptedPrivacyValue:
        return EncryptedPrivacyValue(
            ciphertext=self._encrypt(value, context),
            key_id="test-key",
            algorithm="AES-256-GCM",
        )

    def decrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        algorithm: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> bytes:
        assert key_id == "test-key" and algorithm == "AES-256-GCM"
        return self._decrypt(value, context)

    def start_encryption(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> ObjectArtifactEncryption:
        return ObjectArtifactEncryption(
            key_id="test-key", algorithm="AES-256-GCM"
        )

    def encrypt_chunk(
        self,
        value: bytes,
        *,
        encryption: ObjectArtifactEncryption,
        context: Mapping[str, str] | None = None,
    ) -> bytes:
        return self._encrypt(value, context)

    def decrypt_chunk(
        self,
        value: bytes,
        *,
        encryption: ObjectArtifactEncryption,
        context: Mapping[str, str] | None = None,
    ) -> bytes:
        return self._decrypt(value, context)


async def target(context: TaskTargetContext) -> object:
    return context.input_value


def file_task() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="registered-input", version="1"),
        input=TaskInputContract.file(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agent.toml"),
        run=TaskRunPolicy.queued("q", idempotency=IdempotencyMode.NONE),
        privacy=TaskPrivacyPolicy(
            input=PrivacyAction.ENCRYPT,
            files=PrivacyAction.REDACT,
            file_bytes=PrivacyAction.ENCRYPT,
            raw_retention_days=1,
        ),
        artifact=TaskArtifactPolicy.references_only(retention_days=1),
    )


class TriggerPreparationE2ETest(IsolatedAsyncioTestCase):
    async def test_restart_restore_and_retention_keep_revision_bytes(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "private.txt"
            source.write_bytes(b"registered private bytes")
            clock: list[datetime] = [NOW]
            tasks = InMemoryTaskStore(clock=lambda: clock[0])
            queue = MemoryTaskSubmissionParticipant(tasks)
            ownership = tasks._artifact_ownership
            cipher = ContextCipher()
            backend = ObjectArtifactStore(
                FakeObjectClient(),
                cipher=cipher,
                policy=ObjectArtifactStorePolicy(
                    raw_storage_allowed=True, retention_days=1, max_bytes=4096
                ),
            )
            client = TaskClient(
                tasks,
                target=target,
                queue=queue,
                owner_scope=OWNER.value,
                execution_deployment_id="deployment",
                encryption_provider=cipher,
                raw_storage_allowed=True,
                artifact_store=backend,
                execution_roots=(root,),
                clock=lambda: clock[0],
            )
            await configure_fixture_deployment(client, file_task(), root)
            store = InMemoryTriggerStore(OWNER, clock=lambda: clock[0])
            admission = MemoryTriggerAdmissionStore(store, queue, ownership)
            preparation = TriggerPreparationService(
                client, admission, ownership, cipher
            )
            registration = TriggerRegistrationService(preparation)
            configuration = TriggerConfiguration(
                name="daily",
                task_ref="task.toml",
                schedule=IntervalTrigger(every_seconds=60, start_at=NOW),
                input=TriggerInput(
                    value={
                        "source_kind": "local_path",
                        "reference": "private.txt",
                        "mime_type": "text/plain",
                    }
                ),
            )
            prepared = await preparation.prepare_registration(
                configuration, file_task()
            )
            assert not tasks._runs
            assert b"private.txt" not in prepared.registration.input.ciphertext
            applied = await registration.apply(
                prepared, expected_generation=None
            )
            assert applied.outcome == TriggerCommitOutcome.COMMITTED
            assert applied.snapshot is not None
            assert ownership._revisions and not ownership._staging
            source.unlink()
            # A fresh service restores encrypted file records and validates
            # backend bytes; the original local path no longer exists.
            fresh = TriggerPreparationService(
                client, admission, ownership, cipher
            )
            first = await fresh.prepare_admission(
                plan_admission(applied.snapshot, clock[0])
            )
            committed = await admission.admit(first)
            assert committed.outcome == TriggerCommitOutcome.COMMITTED
            first_ref = first.submissions[0].artifacts[0].ref
            with await backend.open(first_ref) as stream:
                assert stream.read() == b"registered private bytes"
            assert first_ref.artifact_id != prepared.objects[0].ref.artifact_id
            run_id = first.submissions[0].run_id
            retention = TaskRetentionService(tasks, {first_ref.store: backend})
            assert not (
                await retention.enforce_run(
                    run_id, now=NOW + timedelta(days=2)
                )
            ).results
            await tasks.transition_run(
                run_id,
                from_states={TaskRunState.QUEUED},
                to_state=TaskRunState.EXPIRED,
                reason="test",
            )
            assert (
                len(
                    (
                        await retention.enforce_run(
                            run_id, now=NOW + timedelta(days=2)
                        )
                    ).results
                )
                == 1
            )
            assert (await backend.stat(first_ref)).size_bytes == 24
            clock[0] += timedelta(seconds=60)
            assert committed.snapshot is not None
            second = await fresh.prepare_admission(
                plan_admission(committed.snapshot, clock[0])
            )
            assert (
                await admission.admit(second)
            ).outcome == TriggerCommitOutcome.COMMITTED
            history = await store.occurrences("daily", limit=1)
            assert len(history.items) == 1 and history.next_cursor is not None
            assert (
                len(
                    (
                        await store.occurrences(
                            "daily", cursor=history.next_cursor
                        )
                    ).items
                )
                == 1
            )
            second_ref = second.submissions[0].artifacts[0].ref
            assert second_ref.storage_key == first_ref.storage_key
            assert second_ref.artifact_id != first_ref.artifact_id
            current = await store.inspect("daily")
            assert current is not None
            await store.set_enabled(
                "daily",
                enabled=False,
                expected_generation=current.state.generation,
            )
            assert ownership._revisions
            duplicate = await registration.apply(
                prepared, expected_generation=None
            )
            assert duplicate.snapshot == applied.snapshot
            conflicting = replace(
                prepared,
                staging=replace(
                    prepared.staging,
                    recovery_id="00000000-0000-0000-0000-000000000001",
                ),
            )
            conflict = await registration.apply(
                conflicting, expected_generation=999
            )
            assert conflict.error_code == TriggerErrorCode.CONFLICT
            assert conflict.error_path == "generation"
