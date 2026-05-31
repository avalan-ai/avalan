from .context import TaskInputFile
from .definition import IdempotencyMode, TaskDefinition
from .privacy import HmacProvider, TaskKeyMaterial, TaskKeyPurpose
from .store import (
    TaskSnapshotMetadata,
    empty_snapshot_metadata,
    freeze_snapshot_metadata,
)

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from hmac import new as hmac_new
from json import dumps
from math import isfinite

_JSON_SEPARATORS = (",", ":")


class TaskIdempotencyError(ValueError):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskIdempotencyDigest:
    algorithm: str
    digest: str
    key_id: str

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.algorithm, "algorithm")
        _assert_non_empty_string(self.digest, "digest")
        _assert_non_empty_string(self.key_id, "key_id")

    def as_dict(self) -> dict[str, str]:
        return {
            "algorithm": self.algorithm,
            "digest": self.digest,
            "key_id": self.key_id,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskIdempotencyIdentity:
    identity_key: str
    task_name: str
    task_version: str
    spec_hash: str
    owner_scope: TaskIdempotencyDigest
    strategy: IdempotencyMode
    window: TaskIdempotencyDigest | None = None
    input: TaskIdempotencyDigest | None = None
    files: TaskIdempotencyDigest | None = None
    custom: TaskIdempotencyDigest | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.identity_key, "identity_key")
        _assert_non_empty_string(self.task_name, "task_name")
        _assert_non_empty_string(self.task_version, "task_version")
        _assert_non_empty_string(self.spec_hash, "spec_hash")
        assert isinstance(self.owner_scope, TaskIdempotencyDigest)
        assert isinstance(self.strategy, IdempotencyMode)
        assert self.strategy != IdempotencyMode.NONE
        if self.window is not None:
            assert isinstance(self.window, TaskIdempotencyDigest)
        if self.input is not None:
            assert isinstance(self.input, TaskIdempotencyDigest)
        if self.files is not None:
            assert isinstance(self.files, TaskIdempotencyDigest)
        if self.custom is not None:
            assert isinstance(self.custom, TaskIdempotencyDigest)
        if self.strategy == IdempotencyMode.INPUT_HASH:
            assert self.input is not None
            assert self.files is None
            assert self.custom is None
        if self.strategy == IdempotencyMode.INPUT_AND_FILES_HASH:
            assert self.input is not None
            assert self.files is not None
            assert self.custom is None
        if self.strategy == IdempotencyMode.CUSTOM:
            assert self.input is None
            assert self.files is None
            assert self.custom is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "custom": self.custom.as_dict() if self.custom else None,
            "files": self.files.as_dict() if self.files else None,
            "identity_key": self.identity_key,
            "input": self.input.as_dict() if self.input else None,
            "owner_scope": self.owner_scope.as_dict(),
            "spec_hash": self.spec_hash,
            "strategy": self.strategy.value,
            "task_name": self.task_name,
            "task_version": self.task_version,
            "window": self.window.as_dict() if self.window else None,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskIdempotencyReservation:
    identity: TaskIdempotencyIdentity
    run_id: str
    created_at: datetime
    expires_at: datetime | None = None
    metadata: TaskSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        assert isinstance(self.identity, TaskIdempotencyIdentity)
        _assert_non_empty_string(self.run_id, "run_id")
        assert isinstance(self.created_at, datetime)
        if self.expires_at is not None:
            assert isinstance(self.expires_at, datetime)
            assert self.expires_at > self.created_at
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskIdempotencyReservationResult:
    reservation: TaskIdempotencyReservation
    created: bool

    def __post_init__(self) -> None:
        assert isinstance(self.reservation, TaskIdempotencyReservation)
        assert isinstance(self.created, bool)


def task_idempotency_identity(
    definition: TaskDefinition,
    *,
    definition_hash: str,
    input_value: object = None,
    files: tuple[TaskInputFile, ...] = (),
    owner_scope: object,
    hmac_provider: HmacProvider,
    window: object = None,
) -> TaskIdempotencyIdentity | None:
    assert isinstance(definition, TaskDefinition)
    _assert_non_empty_string(definition_hash, "definition_hash")
    assert isinstance(files, tuple)
    for file in files:
        assert isinstance(file, TaskInputFile)
    strategy = definition.run.idempotency
    if strategy == IdempotencyMode.NONE:
        return None
    if hmac_provider is None:
        raise TaskIdempotencyError("idempotency HMAC key is unavailable")
    key = hmac_provider.hmac_key(purpose=TaskKeyPurpose.IDEMPOTENCY)
    owner_digest = _digest(owner_scope, key=key)
    window_digest = _digest(window, key=key) if window is not None else None
    input_digest = None
    files_digest = None
    custom_digest = None
    match strategy:
        case IdempotencyMode.INPUT_HASH:
            input_digest = _digest(input_value, key=key)
        case IdempotencyMode.INPUT_AND_FILES_HASH:
            input_digest = _digest(input_value, key=key)
            files_digest = _digest(_file_identity(files), key=key)
        case IdempotencyMode.CUSTOM:
            custom_digest = _digest(
                _custom_key_value(definition, input_value), key=key
            )
    identity_payload = {
        "custom": custom_digest.as_dict() if custom_digest else None,
        "files": files_digest.as_dict() if files_digest else None,
        "input": input_digest.as_dict() if input_digest else None,
        "owner_scope": owner_digest.as_dict(),
        "spec_hash": definition_hash,
        "strategy": strategy.value,
        "task_name": definition.task.name,
        "task_version": definition.task.version,
        "window": window_digest.as_dict() if window_digest else None,
    }
    identity_key = sha256(_canonical_bytes(identity_payload)).hexdigest()
    return TaskIdempotencyIdentity(
        identity_key=identity_key,
        task_name=definition.task.name,
        task_version=definition.task.version,
        spec_hash=definition_hash,
        owner_scope=owner_digest,
        strategy=strategy,
        window=window_digest,
        input=input_digest,
        files=files_digest,
        custom=custom_digest,
    )


def _digest(value: object, *, key: TaskKeyMaterial) -> TaskIdempotencyDigest:
    payload = _canonical_bytes(value)
    return TaskIdempotencyDigest(
        algorithm=key.algorithm,
        digest=hmac_new(key.secret, payload, sha256).hexdigest(),
        key_id=key.key_id,
    )


def _file_identity(files: tuple[TaskInputFile, ...]) -> tuple[object, ...]:
    return tuple(file.summary() for file in files)


def _custom_key_value(
    definition: TaskDefinition,
    input_value: object,
) -> object:
    path = definition.run.idempotency_key_path
    if path is None:
        raise TaskIdempotencyError("custom idempotency key path is missing")
    value: object = {"input": input_value}
    for part in path.split("."):
        if not part:
            raise TaskIdempotencyError(
                "custom idempotency key path is invalid"
            )
        if not isinstance(value, Mapping) or part not in value:
            raise TaskIdempotencyError("custom idempotency key was not found")
        value = value[part]
    return value


def _canonical_bytes(value: object) -> bytes:
    return dumps(
        _canonical_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=_JSON_SEPARATORS,
        sort_keys=True,
    ).encode("utf-8")


def _canonical_value(value: object) -> object:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TaskIdempotencyError(
                    "idempotency value contains a non-string key"
                )
            normalized[key] = _canonical_value(item)
        return normalized
    if isinstance(value, list | tuple):
        return [_canonical_value(item) for item in value]
    raise TaskIdempotencyError("idempotency value is not JSON-compatible")


def _assert_non_empty_string(value: str | None, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"
