"""Read the shared task submission HMAC authority from operator settings."""

from ..task.privacy import HmacProvider, TaskKeyMaterial, TaskKeyPurpose

from base64 import b64decode
from binascii import Error as BinasciiError
from dataclasses import dataclass, field
from os import environ


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskCliHmacProvider:
    key_id: str
    secret: bytes = field(repr=False)
    algorithm: str = "hmac-sha256"

    def __post_init__(self) -> None:
        assert isinstance(self.key_id, str) and self.key_id.strip()
        assert isinstance(self.secret, bytes) and self.secret
        assert isinstance(self.algorithm, str) and self.algorithm.strip()

    def hmac_key(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
    ) -> TaskKeyMaterial:
        assert isinstance(purpose, TaskKeyPurpose)
        if key_id is not None:
            assert isinstance(key_id, str) and key_id.strip()
        return TaskKeyMaterial(
            key_id=key_id or self.key_id,
            algorithm=self.algorithm,
            secret=self.secret,
        )


def task_hmac_provider() -> HmacProvider | None:
    key_id = environ.get("AVALAN_TASK_HMAC_KEY_ID")
    key_b64 = environ.get("AVALAN_TASK_HMAC_KEY_B64")
    if not (
        isinstance(key_id, str)
        and key_id.strip()
        and isinstance(key_b64, str)
        and key_b64.strip()
    ):
        return None
    try:
        secret = b64decode(key_b64.strip(), validate=True)
    except (BinasciiError, ValueError):
        return None
    return TaskCliHmacProvider(key_id=key_id, secret=secret)
