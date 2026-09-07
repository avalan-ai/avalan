from collections.abc import Mapping
from dataclasses import replace
from typing import cast
from unittest import TestCase
from unittest.mock import patch

from avalan.task.encryption import (
    AesGcmTaskCipher,
    TaskArtifactCipher,
    TaskEncryptionError,
)
from avalan.task.privacy import TaskKeyPurpose


class TaskEncryptionTest(TestCase):
    def test_authenticated_encryption_and_artifact_adapter(self) -> None:
        cipher = AesGcmTaskCipher(key_id="host-key", key=b"k" * 32)
        context = {"physical_id": "object-1", "owner": "owner"}
        encrypted = cipher.encrypt(
            b"secret", purpose=TaskKeyPurpose.RAW_VALUE, context=context
        )
        second = cipher.encrypt(
            b"secret", purpose=TaskKeyPurpose.RAW_VALUE, context=context
        )
        self.assertNotEqual(encrypted.ciphertext, second.ciphertext)
        self.assertNotIn(b"secret", encrypted.ciphertext)
        self.assertEqual(
            cipher.decrypt(
                encrypted.ciphertext,
                purpose=TaskKeyPurpose.RAW_VALUE,
                key_id="host-key",
                algorithm=encrypted.algorithm,
                context=context,
            ),
            b"secret",
        )
        adapter = TaskArtifactCipher(cipher)
        artifact = adapter.encrypt(
            b"", purpose=TaskKeyPurpose.ARTIFACT_CONTENT
        )
        self.assertEqual(
            adapter.decrypt(artifact, purpose=TaskKeyPurpose.ARTIFACT_CONTENT),
            b"",
        )
        for changed, purpose, ctx in (
            (
                replace(
                    encrypted,
                    ciphertext=encrypted.ciphertext[:-1]
                    + bytes([encrypted.ciphertext[-1] ^ 1]),
                ),
                TaskKeyPurpose.RAW_VALUE,
                context,
            ),
            (encrypted, TaskKeyPurpose.ARTIFACT_CONTENT, context),
            (encrypted, TaskKeyPurpose.RAW_VALUE, {"physical_id": "other"}),
        ):
            with self.assertRaises(TaskEncryptionError):
                adapter.decrypt(changed, purpose=purpose, context=ctx)
        with self.assertRaises(TaskEncryptionError):
            adapter.decrypt(
                replace(encrypted, key_id="other"),
                purpose=TaskKeyPurpose.RAW_VALUE,
                context=context,
            )
        with self.assertRaises(TaskEncryptionError):
            cipher.decrypt(
                b"short", purpose=TaskKeyPurpose.RAW_VALUE, algorithm="bad"
            )

    def test_keys_and_optional_dependency_are_checked(self) -> None:
        for name, key in (("", b"k" * 32), ("key", b"short")):
            with self.assertRaises(TaskEncryptionError):
                AesGcmTaskCipher(key_id=name, key=key)
        with patch(
            "avalan.task.encryption.import_module", side_effect=ImportError()
        ):
            with self.assertRaisesRegex(TaskEncryptionError, "dependency"):
                AesGcmTaskCipher(key_id="key", key=b"k" * 32)

    def test_untyped_boundary_rejects_invalid_plaintext_purpose_context(
        self,
    ) -> None:
        cipher = AesGcmTaskCipher(key_id="key", key=b"k" * 32)
        with self.assertRaisesRegex(TaskEncryptionError, "value"):
            cipher.encrypt(
                cast(bytes, "private"), purpose=TaskKeyPurpose.RAW_VALUE
            )
        with self.assertRaisesRegex(TaskEncryptionError, "purpose"):
            cipher.encrypt(
                b"private", purpose=cast(TaskKeyPurpose, "raw_value")
            )
        with self.assertRaisesRegex(TaskEncryptionError, "context"):
            cipher.encrypt(
                b"private",
                purpose=TaskKeyPurpose.RAW_VALUE,
                context=cast(Mapping[str, str], {"owner": 1}),
            )
