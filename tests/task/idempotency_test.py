from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import cast
from unittest import TestCase, main

from avalan.task import (
    HmacProvider,
    IdempotencyMode,
    RunMode,
    TaskDefinition,
    TaskExecutionTarget,
    TaskIdempotencyError,
    TaskInputContract,
    TaskInputFile,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskMetadata,
    TaskOutputContract,
    TaskRunPolicy,
    task_idempotency_identity,
)


class StaticHmacProvider:
    def hmac_key(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
    ) -> TaskKeyMaterial:
        return TaskKeyMaterial(
            key_id=key_id or purpose.value,
            algorithm="hmac-sha256",
            secret=b"idempotency-secret",
        )


def definition(
    mode: IdempotencyMode = IdempotencyMode.INPUT_AND_FILES_HASH,
    *,
    version: str = "1",
) -> TaskDefinition:
    if mode == IdempotencyMode.CUSTOM:
        run = TaskRunPolicy(
            mode=RunMode.QUEUE,
            queue="default",
            idempotency=IdempotencyMode.CUSTOM,
            idempotency_key_path="input.request_id",
        )
    else:
        run = TaskRunPolicy.queued("default", idempotency=mode)
    return TaskDefinition(
        task=TaskMetadata(name="classify", version=version),
        input=TaskInputContract.object(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/classify.toml"),
        run=run,
    )


class TaskIdempotencyIdentityTest(TestCase):
    def test_input_and_file_identity_uses_keyed_hmacs(self) -> None:
        identity = task_idempotency_identity(
            definition(),
            definition_hash="spec-hash",
            input_value={"email": "private@example.com", "answer": 42},
            files=(
                TaskInputFile(
                    logical_path="uploads/private.pdf",
                    media_type="application/pdf",
                    size_bytes=100,
                    metadata={"filename": "private.pdf"},
                ),
            ),
            owner_scope="customer-123",
            hmac_provider=StaticHmacProvider(),
            window="2026-01-01T00:00Z",
        )

        self.assertIsNotNone(identity)
        assert identity is not None
        self.assertEqual(
            identity.strategy, IdempotencyMode.INPUT_AND_FILES_HASH
        )
        self.assertIsNotNone(identity.input)
        self.assertIsNotNone(identity.files)
        self.assertIsNotNone(identity.window)
        self.assertIsNone(identity.custom)
        self.assertEqual(identity.owner_scope.key_id, "idempotency")
        self.assertEqual(len(identity.identity_key), 64)
        rendered = str(identity.as_dict())
        self.assertNotIn("private@example.com", rendered)
        self.assertNotIn("uploads/private.pdf", rendered)
        self.assertNotIn("customer-123", rendered)

    def test_same_semantic_identity_is_stable(self) -> None:
        first = task_idempotency_identity(
            definition(IdempotencyMode.INPUT_HASH),
            definition_hash="spec-hash",
            input_value={"b": 2, "a": 1.5},
            owner_scope="owner",
            hmac_provider=StaticHmacProvider(),
        )
        second = task_idempotency_identity(
            definition(IdempotencyMode.INPUT_HASH),
            definition_hash="spec-hash",
            input_value={"a": 1.5, "b": 2},
            owner_scope="owner",
            hmac_provider=StaticHmacProvider(),
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None
        assert second is not None
        self.assertEqual(first.identity_key, second.identity_key)

    def test_task_version_owner_window_and_files_change_identity(self) -> None:
        base = task_idempotency_identity(
            definition(),
            definition_hash="spec-hash",
            input_value={"a": 1},
            files=(TaskInputFile(logical_path="a.txt"),),
            owner_scope="owner-1",
            hmac_provider=StaticHmacProvider(),
            window="window-1",
        )
        different_owner = task_idempotency_identity(
            definition(),
            definition_hash="spec-hash",
            input_value={"a": 1},
            files=(TaskInputFile(logical_path="a.txt"),),
            owner_scope="owner-2",
            hmac_provider=StaticHmacProvider(),
            window="window-1",
        )
        different_window = task_idempotency_identity(
            definition(),
            definition_hash="spec-hash",
            input_value={"a": 1},
            files=(TaskInputFile(logical_path="a.txt"),),
            owner_scope="owner-1",
            hmac_provider=StaticHmacProvider(),
            window="window-2",
        )
        different_file = task_idempotency_identity(
            definition(),
            definition_hash="spec-hash",
            input_value={"a": 1},
            files=(TaskInputFile(logical_path="b.txt"),),
            owner_scope="owner-1",
            hmac_provider=StaticHmacProvider(),
            window="window-1",
        )
        different_spec_hash = task_idempotency_identity(
            definition(),
            definition_hash="other-spec-hash",
            input_value={"a": 1},
            files=(TaskInputFile(logical_path="a.txt"),),
            owner_scope="owner-1",
            hmac_provider=StaticHmacProvider(),
            window="window-1",
        )
        different_task_version = task_idempotency_identity(
            definition(version="2"),
            definition_hash="spec-hash",
            input_value={"a": 1},
            files=(TaskInputFile(logical_path="a.txt"),),
            owner_scope="owner-1",
            hmac_provider=StaticHmacProvider(),
            window="window-1",
        )
        different_strategy = task_idempotency_identity(
            definition(IdempotencyMode.INPUT_HASH),
            definition_hash="spec-hash",
            input_value={"a": 1},
            owner_scope="owner-1",
            hmac_provider=StaticHmacProvider(),
            window="window-1",
        )

        self.assertIsNotNone(base)
        self.assertIsNotNone(different_owner)
        self.assertIsNotNone(different_window)
        self.assertIsNotNone(different_file)
        self.assertIsNotNone(different_spec_hash)
        self.assertIsNotNone(different_task_version)
        self.assertIsNotNone(different_strategy)
        assert base is not None
        assert different_owner is not None
        assert different_window is not None
        assert different_file is not None
        assert different_spec_hash is not None
        assert different_task_version is not None
        assert different_strategy is not None
        self.assertNotEqual(base.identity_key, different_owner.identity_key)
        self.assertNotEqual(base.identity_key, different_window.identity_key)
        self.assertNotEqual(base.identity_key, different_file.identity_key)
        self.assertNotEqual(
            base.identity_key, different_spec_hash.identity_key
        )
        self.assertNotEqual(
            base.identity_key, different_task_version.identity_key
        )
        self.assertNotEqual(base.identity_key, different_strategy.identity_key)

    def test_file_identity_redacts_hostile_metadata(self) -> None:
        identity = task_idempotency_identity(
            definition(),
            definition_hash="spec-hash",
            input_value={"a": 1},
            files=(
                TaskInputFile(
                    logical_path="provider:openai:provider_file_id",
                    media_type="application/pdf",
                    size_bytes=100,
                    metadata={
                        "filename": "private.pdf",
                        "url": "https://private.example.test/raw",
                    },
                ),
                TaskInputFile(
                    logical_path="artifact:artifact-1",
                    media_type="text/plain",
                    metadata={
                        "identity": {
                            "algorithm": "hmac-sha256",
                            "digest": "file-identity-hmac",
                        }
                    },
                ),
            ),
            owner_scope="owner",
            hmac_provider=StaticHmacProvider(),
        )

        self.assertIsNotNone(identity)
        assert identity is not None
        rendered = str(identity.as_dict())
        self.assertNotIn("private.pdf", rendered)
        self.assertNotIn("private.example", rendered)
        self.assertNotIn("file-identity-hmac", rendered)

    def test_custom_identity_hashes_selected_value_only(self) -> None:
        task = definition(IdempotencyMode.CUSTOM)

        first = task_idempotency_identity(
            task,
            definition_hash="spec-hash",
            input_value={"request_id": "same", "body": "first secret"},
            owner_scope="owner",
            hmac_provider=StaticHmacProvider(),
        )
        second = task_idempotency_identity(
            task,
            definition_hash="spec-hash",
            input_value={"request_id": "same", "body": "second secret"},
            owner_scope="owner",
            hmac_provider=StaticHmacProvider(),
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None
        assert second is not None
        self.assertEqual(first.identity_key, second.identity_key)
        self.assertIsNotNone(first.custom)
        self.assertNotIn("same", str(first.as_dict()))

    def test_none_mode_returns_no_identity(self) -> None:
        identity = task_idempotency_identity(
            definition(IdempotencyMode.NONE),
            definition_hash="spec-hash",
            input_value={"a": 1},
            owner_scope="owner",
            hmac_provider=StaticHmacProvider(),
        )

        self.assertIsNone(identity)

    def test_invalid_values_fail_without_leaking_payloads(self) -> None:
        with self.assertRaises(TaskIdempotencyError) as missing_key:
            task_idempotency_identity(
                definition(IdempotencyMode.CUSTOM),
                definition_hash="spec-hash",
                input_value={"secret": "private"},
                owner_scope="owner",
                hmac_provider=StaticHmacProvider(),
            )
        self.assertNotIn("private", str(missing_key.exception))

        with self.assertRaises(TaskIdempotencyError) as bad_mapping:
            task_idempotency_identity(
                definition(IdempotencyMode.INPUT_HASH),
                definition_hash="spec-hash",
                input_value=cast(Mapping[str, object], {1: "private"}),
                owner_scope="owner",
                hmac_provider=StaticHmacProvider(),
            )
        self.assertNotIn("private", str(bad_mapping.exception))

        missing_path = definition(IdempotencyMode.CUSTOM)
        object.__setattr__(missing_path.run, "idempotency_key_path", None)
        with self.assertRaises(TaskIdempotencyError):
            task_idempotency_identity(
                missing_path,
                definition_hash="spec-hash",
                input_value={"request_id": "private"},
                owner_scope="owner",
                hmac_provider=StaticHmacProvider(),
            )

        invalid_path = definition(IdempotencyMode.CUSTOM)
        object.__setattr__(
            invalid_path.run,
            "idempotency_key_path",
            "input.",
        )
        with self.assertRaises(TaskIdempotencyError):
            task_idempotency_identity(
                invalid_path,
                definition_hash="spec-hash",
                input_value={"request_id": "private"},
                owner_scope="owner",
                hmac_provider=StaticHmacProvider(),
            )

        with self.assertRaises(TaskIdempotencyError):
            task_idempotency_identity(
                definition(IdempotencyMode.INPUT_HASH),
                definition_hash="spec-hash",
                input_value=float("nan"),
                owner_scope="owner",
                hmac_provider=StaticHmacProvider(),
            )

        with self.assertRaises(TaskIdempotencyError):
            task_idempotency_identity(
                definition(IdempotencyMode.INPUT_HASH),
                definition_hash="spec-hash",
                input_value={"safe": True},
                owner_scope="owner",
                hmac_provider=cast(HmacProvider, None),
            )

    def test_identity_dataclasses_are_frozen(self) -> None:
        identity = task_idempotency_identity(
            definition(IdempotencyMode.INPUT_HASH),
            definition_hash="spec-hash",
            input_value={"a": 1},
            owner_scope="owner",
            hmac_provider=StaticHmacProvider(),
        )

        self.assertIsNotNone(identity)
        assert identity is not None
        with self.assertRaises(FrozenInstanceError):
            cast(object, setattr(identity, "identity_key", "changed"))


if __name__ == "__main__":
    main()
