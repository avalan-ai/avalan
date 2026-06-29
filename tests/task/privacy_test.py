from collections.abc import Mapping
from math import inf
from unittest import TestCase, main

from avalan.task import (
    DROPPED_MARKER,
    ENCRYPTED_MARKER,
    HASHED_MARKER,
    REDACTED_MARKER,
    STORED_ENVELOPE_MARKER,
    STORED_MARKER,
    EncryptedPrivacyValue,
    PrivacyAction,
    PrivacyField,
    PrivacySanitizationError,
    PrivacySanitizer,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskPrivacyPolicy,
    decrypt_encrypted_privacy_value,
    privacy_policy_fields,
    privacy_policy_hash_fields,
    privacy_policy_raw_fields,
    privacy_policy_store_fields,
    privacy_policy_with_defaults,
)


class DeterministicHmacProvider:
    def hmac_key(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
    ) -> TaskKeyMaterial:
        self.purpose = purpose
        return TaskKeyMaterial(
            key_id=key_id or "hmac-v1",
            algorithm="hmac-sha256",
            secret=b"test-hmac-key",
        )


class DeterministicEncryptionProvider:
    def encrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> EncryptedPrivacyValue:
        self.value = value
        self.purpose = purpose
        return EncryptedPrivacyValue(
            ciphertext=b"encrypted:" + value,
            key_id=key_id or "enc-v1",
            algorithm="test-aead",
            metadata=context,
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
        self.decrypt_purpose = purpose
        self.decrypt_key_id = key_id
        self.decrypt_algorithm = algorithm
        self.decrypt_context = context
        prefix = b"encrypted:"
        assert value.startswith(prefix)
        return value[len(prefix) :]


class SecretObject:
    def __str__(self) -> str:
        return "private prompt with /Users/person/secret.pdf"

    def __repr__(self) -> str:
        return "raw model output"


class PrivacyTest(TestCase):
    def test_policy_defaults_inherit_unspecified_keys(self) -> None:
        policy = privacy_policy_with_defaults(
            {
                "input": PrivacyAction.REDACT,
                "output": "encrypt",
                "raw_retention_days": 7,
            }
        )

        self.assertEqual(policy.input, PrivacyAction.REDACT)
        self.assertEqual(policy.prompt, PrivacyAction.REDACT)
        self.assertEqual(policy.output, PrivacyAction.ENCRYPT)
        self.assertEqual(policy.file_bytes, PrivacyAction.DROP)
        self.assertEqual(policy.token_text, PrivacyAction.DROP)
        self.assertEqual(policy.tool_arguments, PrivacyAction.REDACT)
        self.assertEqual(policy.tool_results, PrivacyAction.REDACT)
        self.assertEqual(policy.events, PrivacyAction.REDACT)
        self.assertEqual(policy.errors, PrivacyAction.REDACT)
        self.assertEqual(policy.raw_retention_days, 7)
        self.assertEqual(
            privacy_policy_with_defaults(None), TaskPrivacyPolicy()
        )

    def test_policy_field_helpers_are_ordered_and_action_specific(
        self,
    ) -> None:
        policy = TaskPrivacyPolicy(
            input=PrivacyAction.HASH,
            output=PrivacyAction.ENCRYPT,
            tool_results=PrivacyAction.STORE,
            raw_retention_days=7,
        )

        self.assertEqual(
            tuple(privacy_policy_fields(policy)),
            (
                "input",
                "prompt",
                "output",
                "files",
                "file_bytes",
                "token_text",
                "tool_arguments",
                "tool_results",
                "events",
                "errors",
            ),
        )
        self.assertEqual(
            privacy_policy_hash_fields(policy),
            ("input", "files"),
        )
        self.assertEqual(
            privacy_policy_raw_fields(policy),
            ("output", "tool_results"),
        )
        self.assertEqual(
            privacy_policy_store_fields(policy),
            ("tool_results",),
        )

    def test_policy_default_overrides_reject_invalid_shapes(self) -> None:
        with self.assertRaises(AssertionError):
            privacy_policy_with_defaults({"unknown": "drop"})
        with self.assertRaises(AssertionError):
            privacy_policy_with_defaults({"raw_retention_days": "7"})
        with self.assertRaises(ValueError):
            privacy_policy_with_defaults({"input": "unknown"})

    def test_default_policy_drops_or_redacts_sensitive_values(self) -> None:
        sanitizer = PrivacySanitizer()

        self.assertEqual(
            sanitizer.sanitize(PrivacyField.TOKEN_TEXT, "secret token"),
            {"privacy": DROPPED_MARKER},
        )
        self.assertEqual(
            sanitizer.sanitize("prompt", "private prompt"),
            {"privacy": REDACTED_MARKER},
        )
        self.assertEqual(
            sanitizer.sanitize(
                PrivacyField.TOOL_ARGUMENTS,
                {"status": "started", "arguments": {"secret": "raw"}},
            ),
            {"status": "started"},
        )

    def test_hash_uses_keyed_hmac_without_storing_raw_value(self) -> None:
        provider = DeterministicHmacProvider()
        sanitizer = PrivacySanitizer(hmac_provider=provider)

        sanitized = sanitizer.sanitize(
            PrivacyField.INPUT,
            {"query": "private prompt"},
        )

        self.assertEqual(provider.purpose, TaskKeyPurpose.PRIVACY_HASH)
        self.assertEqual(sanitized["privacy"], HASHED_MARKER)
        self.assertEqual(sanitized["algorithm"], "hmac-sha256")
        self.assertEqual(sanitized["key_id"], "hmac-v1")
        self.assertEqual(
            sanitized["digest"],
            "37259ead8f79dbec72fbbd917aafd6b04706b06187cdf2594966c222ded0b8f9",
        )
        self.assertNotIn("private prompt", str(sanitized))

    def test_hash_accepts_binary_values_and_historical_key_ids(self) -> None:
        sanitizer = PrivacySanitizer(hmac_provider=DeterministicHmacProvider())

        bytes_value = sanitizer.sanitize(
            PrivacyField.INPUT,
            b"private bytes",
            key_id="hmac-old",
        )
        bytearray_value = sanitizer.sanitize(
            PrivacyField.INPUT,
            bytearray(b"private bytes"),
            key_id="hmac-old",
        )

        self.assertEqual(bytes_value["key_id"], "hmac-old")
        self.assertEqual(bytearray_value["key_id"], "hmac-old")
        self.assertEqual(bytes_value["digest"], bytearray_value["digest"])

    def test_hash_fails_closed_without_hmac_provider(self) -> None:
        sanitizer = PrivacySanitizer()

        with self.assertRaises(PrivacySanitizationError) as error:
            sanitizer.sanitize(PrivacyField.INPUT, "secret")

        self.assertEqual(
            str(error.exception), "privacy HMAC key is unavailable"
        )

    def test_encrypt_delegates_to_provider_without_plaintext_output(
        self,
    ) -> None:
        provider = DeterministicEncryptionProvider()
        policy = TaskPrivacyPolicy(
            output=PrivacyAction.ENCRYPT,
            raw_retention_days=3,
        )
        sanitizer = PrivacySanitizer(
            policy,
            encryption_provider=provider,
            raw_storage_allowed=True,
        )

        sanitized = sanitizer.sanitize(PrivacyField.OUTPUT, {"answer": "raw"})

        self.assertEqual(provider.purpose, TaskKeyPurpose.RAW_VALUE)
        self.assertEqual(provider.value, b'{"answer":"raw"}')
        self.assertEqual(sanitized["privacy"], ENCRYPTED_MARKER)
        self.assertEqual(sanitized["algorithm"], "test-aead")
        self.assertEqual(sanitized["key_id"], "enc-v1")
        self.assertNotIn("raw", str(sanitized))

    def test_decrypts_encrypted_privacy_value(self) -> None:
        provider = DeterministicEncryptionProvider()
        policy = TaskPrivacyPolicy(
            input=PrivacyAction.ENCRYPT,
            raw_retention_days=3,
        )
        sanitizer = PrivacySanitizer(
            policy,
            encryption_provider=provider,
            raw_storage_allowed=True,
        )

        encrypted = sanitizer.sanitize(
            PrivacyField.INPUT,
            {"prompt": "private", "limit": 2},
            key_id="enc-old",
        )
        encrypted["metadata"] = {"tenant": "safe"}
        decrypted = decrypt_encrypted_privacy_value(
            encrypted,
            decryption_provider=provider,
        )

        self.assertEqual(
            decrypted,
            {"limit": 2, "prompt": "private"},
        )
        self.assertEqual(provider.decrypt_purpose, TaskKeyPurpose.RAW_VALUE)
        self.assertEqual(provider.decrypt_key_id, "enc-old")
        self.assertEqual(provider.decrypt_algorithm, "test-aead")
        self.assertEqual(provider.decrypt_context, {"tenant": "safe"})

    def test_decrypt_rejects_malformed_envelopes_safely(self) -> None:
        provider = DeterministicEncryptionProvider()

        bad_values: tuple[object, ...] = (
            "private",
            {"privacy": "<redacted>"},
            {
                "privacy": ENCRYPTED_MARKER,
                "key_id": "enc",
                "algorithm": "test-aead",
            },
            {
                "privacy": ENCRYPTED_MARKER,
                "ciphertext": "%%% private %%%",
                "key_id": "enc",
                "algorithm": "test-aead",
            },
            {
                "privacy": ENCRYPTED_MARKER,
                "ciphertext": "\u2603",
                "key_id": "enc",
                "algorithm": "test-aead",
            },
            {
                "privacy": ENCRYPTED_MARKER,
                "ciphertext": "ZW5jcnlwdGVkOnByaXZhdGU=",
                "key_id": "",
                "algorithm": "test-aead",
            },
            {
                "privacy": ENCRYPTED_MARKER,
                "ciphertext": "ZW5jcnlwdGVkOnByaXZhdGU=",
                "key_id": "enc",
                "algorithm": "",
            },
            {
                "privacy": ENCRYPTED_MARKER,
                "ciphertext": "ZW5jcnlwdGVkOnByaXZhdGU=",
                "key_id": "enc",
                "algorithm": "test-aead",
                "metadata": "private",
            },
            {
                "privacy": ENCRYPTED_MARKER,
                "ciphertext": "ZW5jcnlwdGVkOnByaXZhdGU=",
                "key_id": "enc",
                "algorithm": "test-aead",
                "metadata": {"safe": object()},
            },
            {
                "privacy": ENCRYPTED_MARKER,
                "ciphertext": "ZW5jcnlwdGVkOm5vdC1qc29u",
                "key_id": "enc",
                "algorithm": "test-aead",
            },
            {
                "privacy": ENCRYPTED_MARKER,
                "ciphertext": "ZW5jcnlwdGVkOv8=",
                "key_id": "enc",
                "algorithm": "test-aead",
            },
        )

        for value in bad_values:
            with self.subTest(value=value):
                with self.assertRaises(PrivacySanitizationError) as error:
                    decrypt_encrypted_privacy_value(
                        value,
                        decryption_provider=provider,
                    )
                self.assertNotIn("private", str(error.exception))

        with self.assertRaises(PrivacySanitizationError) as missing:
            decrypt_encrypted_privacy_value(
                {
                    "privacy": ENCRYPTED_MARKER,
                    "ciphertext": "ZW5jcnlwdGVkOnByaXZhdGU=",
                    "key_id": "enc",
                    "algorithm": "test-aead",
                },
                decryption_provider=object(),
            )
        self.assertEqual(
            str(missing.exception),
            "privacy decryption key is unavailable",
        )

    def test_encrypt_and_store_require_explicit_raw_retention(self) -> None:
        encrypted = PrivacySanitizer(
            TaskPrivacyPolicy(output=PrivacyAction.ENCRYPT),
            encryption_provider=DeterministicEncryptionProvider(),
            raw_storage_allowed=True,
        )
        stored = PrivacySanitizer(
            TaskPrivacyPolicy(output=PrivacyAction.STORE),
            raw_storage_allowed=True,
        )
        disabled = PrivacySanitizer(
            TaskPrivacyPolicy(output=PrivacyAction.STORE, raw_retention_days=1)
        )
        missing_key = PrivacySanitizer(
            TaskPrivacyPolicy(
                output=PrivacyAction.ENCRYPT,
                raw_retention_days=1,
            ),
            raw_storage_allowed=True,
        )

        for sanitizer in (encrypted, stored, disabled):
            with self.subTest(sanitizer=sanitizer):
                with self.assertRaises(PrivacySanitizationError):
                    sanitizer.sanitize(PrivacyField.OUTPUT, "secret")
        with self.assertRaises(PrivacySanitizationError) as error:
            missing_key.sanitize(PrivacyField.OUTPUT, "secret")
        self.assertEqual(
            str(error.exception),
            "privacy encryption key is unavailable",
        )

        retained = PrivacySanitizer(
            TaskPrivacyPolicy(
                output=PrivacyAction.STORE,
                raw_retention_days=1,
            ),
            raw_storage_allowed=True,
        )
        self.assertEqual(
            retained.sanitize(PrivacyField.OUTPUT, {"safe": True}),
            {
                "format": STORED_ENVELOPE_MARKER,
                "privacy": STORED_MARKER,
                "value": {"safe": True},
            },
        )

    def test_store_serializes_json_scalars_and_binary_values(self) -> None:
        sanitizer = PrivacySanitizer(
            TaskPrivacyPolicy(
                output=PrivacyAction.STORE, raw_retention_days=1
            ),
            raw_storage_allowed=True,
        )

        sanitized = sanitizer.sanitize(
            PrivacyField.OUTPUT,
            [1, 1.5, b"raw", bytearray(b"bytes")],
        )

        self.assertEqual(
            sanitized,
            {
                "format": STORED_ENVELOPE_MARKER,
                "privacy": STORED_MARKER,
                "value": [
                    1,
                    1.5,
                    {"encoding": "base64", "value": "cmF3"},
                    {"encoding": "base64", "value": "Ynl0ZXM="},
                ],
            },
        )
        with self.assertRaises(PrivacySanitizationError) as error:
            sanitizer.sanitize(PrivacyField.OUTPUT, {1: "raw"})
        self.assertEqual(
            str(error.exception),
            "privacy value contains a non-string key",
        )

    def test_raw_storage_rejects_adversarial_values_without_repr(
        self,
    ) -> None:
        sanitizer = PrivacySanitizer(
            TaskPrivacyPolicy(
                output=PrivacyAction.STORE, raw_retention_days=1
            ),
            raw_storage_allowed=True,
        )

        with self.assertRaises(PrivacySanitizationError) as nan_error:
            sanitizer.sanitize(PrivacyField.OUTPUT, float("nan"))
        with self.assertRaises(PrivacySanitizationError) as object_error:
            sanitizer.sanitize(PrivacyField.OUTPUT, SecretObject())

        rendered = f"{nan_error.exception} {object_error.exception}"
        self.assertNotIn("private prompt", rendered)
        self.assertNotIn("/Users/person/secret.pdf", rendered)
        self.assertNotIn("raw model output", rendered)

    def test_event_hashing_uses_the_configured_event_action(self) -> None:
        sanitizer = PrivacySanitizer(
            TaskPrivacyPolicy(events=PrivacyAction.HASH),
            hmac_provider=DeterministicHmacProvider(),
        )

        event = sanitizer.sanitize_event(
            "token",
            {"token_text": "private token"},
        )

        self.assertEqual(event["privacy"], HASHED_MARKER)
        self.assertNotIn("private token", str(event))

    def test_recursive_event_sanitization_drops_unknown_fields(self) -> None:
        sanitizer = PrivacySanitizer(
            event_allowlists={"tool": ("count", "status")}
        )
        engine_event = sanitizer.sanitize_event(
            "start",
            {"status": "ok", "prompt": "private prompt"},
        )
        event = sanitizer.sanitize_event(
            "tool",
            {
                "arguments": {"secret": "raw argument"},
                "count": 2,
                "nested": {"secret": "raw result"},
                "status": "ok",
                "timestamp": "2026-05-30T00:00:00Z",
            },
        )
        unknown = sanitizer.sanitize_event(
            "new-provider-event",
            {
                "prompt": "private prompt",
                "status": "ok",
                "token_text": "secret token",
            },
        )

        self.assertEqual(
            engine_event,
            {"event_type": "start", "status": "ok"},
        )
        self.assertEqual(
            event,
            {
                "count": 2,
                "event_type": "tool",
                "status": "ok",
                "timestamp": "2026-05-30T00:00:00Z",
            },
        )
        self.assertEqual(
            unknown,
            {
                "event_type": "new-provider-event",
                "status": "ok",
            },
        )
        self.assertNotIn("private prompt", str(unknown))
        self.assertNotIn("secret token", str(unknown))

    def test_event_redaction_keeps_flow_node_and_token_type(self) -> None:
        sanitizer = PrivacySanitizer()

        event = sanitizer.sanitize_event(
            "token_generated",
            {
                "flow_node": "analyze_pov_1",
                "token": "private token",
                "token_type": "ReasoningToken",
            },
        )

        self.assertEqual(
            event,
            {
                "event_type": "token_generated",
                "flow_node": "analyze_pov_1",
                "token_type": "ReasoningToken",
            },
        )
        self.assertNotIn("private token", str(event))

    def test_event_redaction_keeps_canonical_stream_metadata(self) -> None:
        sanitizer = PrivacySanitizer()

        event = sanitizer.sanitize_event(
            "token_generated",
            {
                "canonical_stream": {
                    "stream_session_id": "stream-1",
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "sequence": 3,
                    "kind": "reasoning.delta",
                    "channel": "reasoning",
                    "visibility": "private",
                    "summary": {"text_delta_length": 12},
                    "text_delta": "private token",
                },
                "token": "private token",
            },
        )

        self.assertEqual(
            event,
            {
                "event_type": "token_generated",
                "canonical_stream": {
                    "stream_session_id": "stream-1",
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "sequence": 3,
                    "kind": "reasoning.delta",
                    "channel": "reasoning",
                    "visibility": "private",
                },
            },
        )
        self.assertNotIn("private token", str(event))

    def test_pipeline_event_redaction_keeps_stage_status_only(self) -> None:
        sanitizer = PrivacySanitizer()

        event = sanitizer.sanitize_event(
            "tool_execution_completed",
            {
                "name": "shell.pipeline",
                "status": "completed",
                "stage_count": 2,
                "stage_statuses": [
                    {
                        "stage_id": "search",
                        "stage_index": 0,
                        "status": "completed",
                        "stdout": "private intermediate stdout",
                        "command": "rg",
                        "cwd": "/Users/person/private-project",
                    },
                    {
                        "stage_id": "count",
                        "stage_index": 1,
                        "status": "completed",
                        "stdout": "private final stdout",
                        "stdin_from": {"step_id": "search"},
                    },
                ],
                "stdout": "private final stdout",
                "stderr": "private diagnostics",
                "intermediate_stdout": "private intermediate stdout",
                "tool_arguments": {"steps": [{"command": "rg"}]},
                "usage": {"metadata": "private shell metadata"},
                "artifacts": [{"path": "/Users/person/private-project/out"}],
                "canonical_stream": {
                    "stream_session_id": "stream-1",
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "sequence": 7,
                    "kind": "tool.result",
                    "channel": "tool",
                    "visibility": "public",
                    "summary": {
                        "stdout": "private intermediate stdout",
                        "tool_arguments": {"steps": [{"command": "rg"}]},
                    },
                    "private_metadata": {
                        "cwd": "/Users/person/private-project"
                    },
                },
            },
        )

        self.assertEqual(
            event,
            {
                "event_type": "tool_execution_completed",
                "name": "shell.pipeline",
                "status": "completed",
                "stage_count": 2,
                "stage_statuses": [
                    {
                        "stage_id": "search",
                        "stage_index": 0,
                        "status": "completed",
                    },
                    {
                        "stage_id": "count",
                        "stage_index": 1,
                        "status": "completed",
                    },
                ],
                "canonical_stream": {
                    "stream_session_id": "stream-1",
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "sequence": 7,
                    "kind": "tool.result",
                    "channel": "tool",
                    "visibility": "public",
                },
            },
        )
        rendered = str(event)
        self.assertNotIn("private intermediate stdout", rendered)
        self.assertNotIn("private final stdout", rendered)
        self.assertNotIn("private shell metadata", rendered)
        self.assertNotIn("/Users/person/private-project", rendered)
        self.assertNotIn("rg", rendered)

    def test_event_redaction_drops_error_text_and_paths(self) -> None:
        sanitizer = PrivacySanitizer()

        event = sanitizer.sanitize_event(
            "model_error",
            {
                "code": "provider_timeout",
                "count": 1,
                "details": {
                    "code": "nested_timeout",
                    "message": "private exception text",
                    "path": "/Users/person/secret.pdf",
                },
                "duration_ms": 1.5,
                "elapsed_ms": inf,
                "message": "private exception text",
                "path": "/Users/person/secret.pdf",
                "retryable": True,
                "stack": "private stack trace",
                "status": [
                    "failed",
                    SecretObject(),
                    {
                        "code": "nested_status",
                        "message": "private nested message",
                    },
                    {"secret": "private nested value"},
                ],
            },
        )

        self.assertEqual(
            event,
            {
                "code": "provider_timeout",
                "count": 1,
                "duration_ms": 1.5,
                "elapsed_ms": REDACTED_MARKER,
                "event_type": "model_error",
                "retryable": True,
                "status": [
                    "failed",
                    REDACTED_MARKER,
                    {"code": "nested_status"},
                    REDACTED_MARKER,
                ],
            },
        )
        self.assertNotIn("private", str(event))
        self.assertNotIn("/Users/person/secret.pdf", str(event))
        self.assertNotIn("message", str(event))
        self.assertNotIn("path", str(event))
        self.assertNotIn("stack", str(event))

    def test_safe_metadata_redaction_is_recursive_and_bounded(self) -> None:
        sanitizer = PrivacySanitizer()

        redacted = sanitizer.sanitize(
            PrivacyField.PROMPT,
            {
                "category": SecretObject(),
                "count": 1,
                "counts": {"count": 2},
                "duration_ms": 1.5,
                "elapsed_ms": inf,
                "status": ["ok", SecretObject()],
                "timestamp": None,
                "type": {"secret": "raw"},
            },
        )

        self.assertEqual(
            redacted,
            {
                "category": REDACTED_MARKER,
                "count": 1,
                "counts": {"count": 2},
                "duration_ms": 1.5,
                "elapsed_ms": REDACTED_MARKER,
                "status": ["ok", REDACTED_MARKER],
                "timestamp": None,
                "type": REDACTED_MARKER,
            },
        )

    def test_unknown_objects_are_never_stringified_or_represented(
        self,
    ) -> None:
        secret = SecretObject()
        sanitizer = PrivacySanitizer(hmac_provider=DeterministicHmacProvider())

        redacted = sanitizer.sanitize(PrivacyField.PROMPT, secret)
        with self.assertRaises(PrivacySanitizationError):
            sanitizer.sanitize(PrivacyField.INPUT, secret)

        rendered = f"{redacted}"
        self.assertEqual(redacted, {"privacy": REDACTED_MARKER})
        self.assertNotIn("private prompt", rendered)
        self.assertNotIn("/Users/person/secret.pdf", rendered)
        self.assertNotIn("raw model output", rendered)

    def test_value_objects_validate_metadata_without_secret_bytes(
        self,
    ) -> None:
        key = TaskKeyMaterial(
            key_id="kid",
            algorithm="hmac-sha256",
            secret=b"secret",
        )
        encrypted = EncryptedPrivacyValue(
            ciphertext=b"ciphertext",
            key_id="enc",
            algorithm="aead",
            metadata={"purpose": "test"},
        )

        self.assertEqual(
            key.metadata(),
            {"algorithm": "hmac-sha256", "key_id": "kid"},
        )
        self.assertEqual(
            encrypted.as_dict(),
            {
                "algorithm": "aead",
                "ciphertext": "Y2lwaGVydGV4dA==",
                "key_id": "enc",
                "metadata": {"purpose": "test"},
                "privacy": ENCRYPTED_MARKER,
            },
        )
        with self.assertRaises(AssertionError):
            TaskKeyMaterial(key_id="", algorithm="hmac-sha256", secret=b"x")
        with self.assertRaises(AssertionError):
            EncryptedPrivacyValue(
                ciphertext=b"",
                key_id="enc",
                algorithm="aead",
            )


if __name__ == "__main__":
    main()
