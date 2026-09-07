"""Encrypt durable task input with a host-managed AES-256-GCM key."""

from .privacy import EncryptedPrivacyValue, TaskKeyPurpose

from collections.abc import Mapping
from importlib import import_module
from json import dumps
from os import urandom
from typing import Protocol, cast


class TaskEncryptionError(ValueError):
    """Expose a safe encryption failure without key or plaintext details."""


class _AesGcm(Protocol):
    def encrypt(
        self, nonce: bytes, data: bytes, associated_data: bytes
    ) -> bytes: ...
    def decrypt(
        self, nonce: bytes, data: bytes, associated_data: bytes
    ) -> bytes: ...


class _AesGcmFactory(Protocol):
    def __call__(self, key: bytes) -> _AesGcm: ...


class AesGcmTaskCipher:
    """Bind authenticated ciphertext to key, purpose and physical context."""

    def __init__(self, *, key_id: str, key: bytes) -> None:
        if (
            not isinstance(key_id, str)
            or not key_id.strip()
            or len(key_id) > 128
        ):
            raise TaskEncryptionError("encryption.key_id")
        if not isinstance(key, bytes) or len(key) != 32:
            raise TaskEncryptionError("encryption.key")
        try:
            module = import_module(
                "cryptography.hazmat.primitives.ciphers.aead"
            )
        except ImportError:
            raise TaskEncryptionError("encryption.dependency") from None
        factory = cast(_AesGcmFactory, module.AESGCM)
        self._cipher = factory(key)
        self._key_id = key_id

    def _aad(
        self,
        purpose: TaskKeyPurpose,
        key_id: str | None,
        context: Mapping[str, str] | None,
    ) -> bytes:
        if key_id is not None and key_id != self._key_id:
            raise TaskEncryptionError("encryption.key_id")
        if not isinstance(purpose, TaskKeyPurpose):
            raise TaskEncryptionError("encryption.purpose")
        if context is not None and not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in context.items()
        ):
            raise TaskEncryptionError("encryption.context")
        return dumps(
            {
                "key_id": self._key_id,
                "purpose": purpose.value,
                "context": dict(context or {}),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()

    def encrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> EncryptedPrivacyValue:
        if not isinstance(value, bytes):
            raise TaskEncryptionError("encryption.value")
        aad = self._aad(purpose, key_id, context)
        nonce = urandom(12)
        return EncryptedPrivacyValue(
            ciphertext=nonce + self._cipher.encrypt(nonce, value, aad),
            key_id=self._key_id,
            algorithm="aes-256-gcm.v1",
            metadata=dict(context) if context else None,
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
        if (
            not isinstance(value, bytes)
            or len(value) < 28
            or algorithm != "aes-256-gcm.v1"
        ):
            raise TaskEncryptionError("encryption.envelope")
        aad = self._aad(purpose, key_id, context)
        try:
            return self._cipher.decrypt(value[:12], value[12:], aad)
        except Exception:
            raise TaskEncryptionError("encryption.authentication") from None


class TaskArtifactCipher:
    """Adapt authenticated encryption to the artifact envelope protocol."""

    def __init__(self, cipher: AesGcmTaskCipher) -> None:
        self._cipher = cipher

    def encrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> EncryptedPrivacyValue:
        return self._cipher.encrypt(
            value, purpose=purpose, key_id=key_id, context=context
        )

    def decrypt(
        self,
        value: EncryptedPrivacyValue,
        *,
        purpose: TaskKeyPurpose,
        context: Mapping[str, str] | None = None,
    ) -> bytes:
        return self._cipher.decrypt(
            value.ciphertext,
            purpose=purpose,
            key_id=value.key_id,
            algorithm=value.algorithm,
            context=context,
        )
