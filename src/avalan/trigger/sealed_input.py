"""Encrypt registered input and its validated materialized file records."""

from ..task.materialization import TaskMaterializedFile
from ..task.privacy import (
    DecryptionProvider,
    EncryptionProvider,
    TaskKeyPurpose,
)
from ..task.runner import (
    TaskExecutableInputFileEntry,
    task_execution_file_entries_from_value,
    task_execution_file_entries_value,
)
from .canonical import plain_value
from .definition import BindingSource, InputBinding, TriggerInput
from .error import TriggerError, TriggerErrorCode
from .records import OwnerScopeId, TriggerSealedInput

from collections.abc import Mapping
from dataclasses import dataclass, field
from json import dumps, loads


@dataclass(frozen=True, slots=True, kw_only=True)
class RegisteredTriggerInput:
    """Retain decrypted input only within a preparation scope."""

    input: TriggerInput = field(repr=False)
    files: tuple[TaskMaterializedFile, ...] = field(default=(), repr=False)


def _context(
    owner: OwnerScopeId, name: str, semantic_hash: str
) -> Mapping[str, str]:
    return {
        "owner_scope": owner.value,
        "trigger_name": name,
        "semantic_hash": semantic_hash,
        "format": "avalan.trigger.input.v1",
    }


def seal_input(
    value: RegisteredTriggerInput,
    provider: EncryptionProvider,
    *,
    owner: OwnerScopeId,
    name: str,
    semantic_hash: str,
) -> TriggerSealedInput:
    """Encrypt one canonical input envelope with owner-bound context."""
    payload = {
        "format": "avalan.trigger.input",
        "version": 1,
        "input": plain_value(value.input.value),
        "bindings": [
            {"path": binding.path, "source": binding.source.value}
            for binding in value.input.bindings
        ],
        "files": plain_value(
            task_execution_file_entries_value(
                tuple(
                    TaskExecutableInputFileEntry(
                        file=file.as_input_file(), materialized_file=file
                    )
                    for file in value.files
                )
            )
        ),
    }
    encrypted = provider.encrypt(
        dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8"),
        purpose=TaskKeyPurpose.RAW_VALUE,
        context=_context(owner, name, semantic_hash),
    )
    return TriggerSealedInput(
        ciphertext=encrypted.ciphertext,
        key_id=encrypted.key_id,
        algorithm=encrypted.algorithm,
        metadata=encrypted.metadata or {},
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate input field")
        value[key] = item
    return value


def unseal_input(
    value: TriggerSealedInput,
    provider: DecryptionProvider,
    *,
    owner: OwnerScopeId,
    name: str,
    semantic_hash: str,
) -> RegisteredTriggerInput:
    """Authenticate context and reject unsupported or malformed envelopes."""
    try:
        plaintext = provider.decrypt(
            value.ciphertext,
            purpose=TaskKeyPurpose.RAW_VALUE,
            key_id=value.key_id,
            algorithm=value.algorithm,
            context=_context(owner, name, semantic_hash),
        )
        payload = loads(plaintext, object_pairs_hook=_unique_object)
        if not isinstance(payload, dict) or set(payload) != {
            "format",
            "version",
            "input",
            "bindings",
            "files",
        }:
            raise ValueError("invalid input envelope")
        if payload["format"] != "avalan.trigger.input" or (
            type(payload["version"]) is not int or payload["version"] != 1
        ):
            raise TriggerError(TriggerErrorCode.UNSUPPORTED_VERSION, "input")
        bindings = payload["bindings"]
        if not isinstance(bindings, list):
            raise ValueError("invalid input bindings")
        parsed: list[InputBinding] = []
        for binding in bindings:
            if not isinstance(binding, dict) or set(binding) != {
                "path",
                "source",
            }:
                raise ValueError("invalid input binding")
            parsed.append(
                InputBinding(
                    path=binding["path"],
                    source=BindingSource(binding["source"]),
                )
            )
        files = task_execution_file_entries_from_value(payload["files"])
        materialized: list[TaskMaterializedFile] = []
        for file in files:
            if file.materialized_file is None:
                raise ValueError("registered artifact is not materialized")
            materialized.append(file.materialized_file)
        return RegisteredTriggerInput(
            input=TriggerInput(value=payload["input"], bindings=tuple(parsed)),
            files=tuple(materialized),
        )
    except TriggerError:
        raise
    except Exception:
        raise TriggerError(
            TriggerErrorCode.INVALID_CONFIG, "input.encrypted"
        ) from None
