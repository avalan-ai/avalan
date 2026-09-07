from .preparation_e2e_test import ContextCipher
from .records_test import OWNER

from dataclasses import replace
from json import dumps

from pytest import raises

from avalan.task.privacy import TaskKeyPurpose
from avalan.trigger.definition import BindingSource, InputBinding, TriggerInput
from avalan.trigger.error import TriggerError, TriggerErrorCode
from avalan.trigger.records import TriggerSealedInput
from avalan.trigger.sealed_input import (
    RegisteredTriggerInput,
    _context,
    seal_input,
    unseal_input,
)


def test_encrypted_input_round_trip_and_strict_envelope() -> None:
    cipher = ContextCipher()
    original = RegisteredTriggerInput(
        input=TriggerInput(
            value={"when": "placeholder"},
            bindings=(
                InputBinding(path="/when", source=BindingSource.SCHEDULED_AT),
            ),
        )
    )
    encrypted = seal_input(
        original, cipher, owner=OWNER, name="daily", semantic_hash="a" * 64
    )
    assert (
        unseal_input(
            encrypted,
            cipher,
            owner=OWNER,
            name="daily",
            semantic_hash="a" * 64,
        )
        == original
    )
    with raises(TriggerError):
        unseal_input(
            encrypted,
            cipher,
            owner=OWNER,
            name="other",
            semantic_hash="a" * 64,
        )
    valid = {
        "format": "avalan.trigger.input",
        "version": 1,
        "input": None,
        "bindings": [],
        "files": [],
    }
    cases: tuple[tuple[object, TriggerErrorCode], ...] = (
        ([], TriggerErrorCode.INVALID_CONFIG),
        ({**valid, "extra": True}, TriggerErrorCode.INVALID_CONFIG),
        ({**valid, "version": 2}, TriggerErrorCode.UNSUPPORTED_VERSION),
        ({**valid, "version": True}, TriggerErrorCode.UNSUPPORTED_VERSION),
        ({**valid, "format": "other"}, TriggerErrorCode.UNSUPPORTED_VERSION),
        ({**valid, "bindings": {}}, TriggerErrorCode.INVALID_CONFIG),
        ({**valid, "bindings": [None]}, TriggerErrorCode.INVALID_CONFIG),
        (
            {**valid, "bindings": [{"path": "/bad", "source": "now"}]},
            TriggerErrorCode.INVALID_CONFIG,
        ),
        (
            {
                **valid,
                "files": [{"file": {"logical_path": "file"}, "order": 0}],
            },
            TriggerErrorCode.INVALID_CONFIG,
        ),
    )
    for payload, code in cases:
        encoded = cipher.encrypt(
            dumps(payload).encode(),
            purpose=TaskKeyPurpose.RAW_VALUE,
            context=_context(OWNER, "daily", "a" * 64),
        )
        value = TriggerSealedInput(
            ciphertext=encoded.ciphertext,
            key_id=encoded.key_id,
            algorithm=encoded.algorithm,
        )
        with raises(TriggerError) as caught:
            unseal_input(
                value,
                cipher,
                owner=OWNER,
                name="daily",
                semantic_hash="a" * 64,
            )
        assert caught.value.code == code
    duplicate = cipher.encrypt(
        b'{"input": 1, "input": 2}',
        purpose=TaskKeyPurpose.RAW_VALUE,
        context=_context(OWNER, "daily", "a" * 64),
    )
    with raises(TriggerError):
        unseal_input(
            replace(encrypted, ciphertext=duplicate.ciphertext),
            cipher,
            owner=OWNER,
            name="daily",
            semantic_hash="a" * 64,
        )
