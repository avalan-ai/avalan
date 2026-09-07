"""Compose the explicitly authorized durable CLI input storage boundary."""

from ..pgsql import PgsqlDatabase
from ..task.artifacts.pgsql import (
    PgsqlArtifactByteStoragePolicy,
    PgsqlArtifactStore,
)
from ..task.encryption import (
    AesGcmTaskCipher,
    TaskArtifactCipher,
    TaskEncryptionError,
)
from ..task.feature_gate import TaskFeature

from argparse import Namespace
from base64 import b64decode
from binascii import Error as Base64Error
from os import environ


def encrypted_input_store(
    args: Namespace, database: PgsqlDatabase
) -> tuple[AesGcmTaskCipher, PgsqlArtifactStore]:
    """Load operator key material without exposing it in diagnostics."""
    if not args.raw_storage_allowed:
        raise TaskEncryptionError("encryption.raw_storage_authorization")
    if (
        isinstance(args.retention_days, bool)
        or not isinstance(args.retention_days, int)
        or not 1 <= args.retention_days <= 3650
    ):
        raise TaskEncryptionError("encryption.retention_days")
    if (
        isinstance(args.max_artifact_bytes, bool)
        or not isinstance(args.max_artifact_bytes, int)
        or not 1 <= args.max_artifact_bytes <= 1073741824
    ):
        raise TaskEncryptionError("encryption.max_artifact_bytes")
    key_id = environ.get("AVALAN_TASK_ENCRYPTION_KEY_ID", "")
    try:
        key = b64decode(
            environ.get("AVALAN_TASK_ENCRYPTION_KEY_B64", ""), validate=True
        )
    except (Base64Error, ValueError):
        raise TaskEncryptionError("encryption.key") from None
    cipher = AesGcmTaskCipher(key_id=key_id, key=key)
    artifacts = PgsqlArtifactStore(
        database,
        cipher=TaskArtifactCipher(cipher),
        policy=PgsqlArtifactByteStoragePolicy(
            raw_storage_allowed=True,
            retention_days=args.retention_days,
            max_bytes=args.max_artifact_bytes,
            enabled_features=(TaskFeature.RAW_STORAGE,),
        ),
    )
    return cipher, artifacts
