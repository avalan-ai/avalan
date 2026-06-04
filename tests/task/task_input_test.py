from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from typing import cast
from unittest import TestCase, main

from avalan.task import (
    TaskDefinition,
    TaskExecutionTarget,
    TaskFileConversionRequest,
    TaskFileDescriptor,
    TaskFileSourceKind,
    TaskInputContract,
    TaskInputFile,
    TaskLimitsPolicy,
    TaskMetadata,
    TaskOutputContract,
    TaskProviderReference,
    TaskProviderReferenceKind,
    TaskRemoteUrlPolicy,
    validate_task_definition,
    validate_task_input,
)


class TaskInputTest(TestCase):
    def test_file_descriptor_helpers_freeze_nested_metadata(self) -> None:
        options = {"quality": "text", "pages": [1, 2]}
        metadata = {"caller": {"tenant": "alpha"}, "tags": ["draft"]}
        conversion = TaskFileConversionRequest(
            name="markdown",
            options=options,
        )
        descriptor = TaskFileDescriptor.local_path(
            "uploads/private.pdf",
            role="source",
            mime_type="application/pdf",
            size_bytes=100,
            sha256="a" * 64,
            conversions=(conversion,),
            metadata=metadata,
        )

        options["quality"] = "image"
        cast(list[int], options["pages"]).append(3)
        cast(dict[str, str], metadata["caller"])["tenant"] = "mutated"
        cast(list[str], metadata["tags"]).append("mutated")

        self.assertEqual(descriptor.source_kind, TaskFileSourceKind.LOCAL_PATH)
        self.assertEqual(conversion.options["quality"], "text")
        self.assertEqual(conversion.options["pages"], (1, 2))
        caller = cast(Mapping[str, object], descriptor.metadata["caller"])
        self.assertEqual(caller["tenant"], "alpha")
        self.assertEqual(descriptor.metadata["tags"], ("draft",))
        with self.assertRaises(TypeError):
            cast(dict[str, object], descriptor.metadata)["added"] = True
        with self.assertRaises(FrozenInstanceError):
            descriptor.reference = "uploads/other.pdf"

    def test_file_descriptor_helpers_cover_source_kinds(self) -> None:
        descriptors = (
            TaskFileDescriptor.local_path("inputs/a.txt"),
            TaskFileDescriptor.remote_url("https://example.test/a.txt"),
            TaskFileDescriptor.artifact("artifact-1"),
            TaskFileDescriptor.inline_bytes("inline-1"),
            TaskFileDescriptor.provider_reference_descriptor(
                "file-openai",
                kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                provider="openai",
            ),
        )

        self.assertEqual(
            [descriptor.source_kind for descriptor in descriptors],
            [
                TaskFileSourceKind.LOCAL_PATH,
                TaskFileSourceKind.REMOTE_URL,
                TaskFileSourceKind.ARTIFACT,
                TaskFileSourceKind.INLINE_BYTES,
                TaskFileSourceKind.PROVIDER_REFERENCE,
            ],
        )

    def test_provider_reference_summary_omits_raw_handle(self) -> None:
        expires_at = datetime.now(UTC) + timedelta(hours=1)
        reference = TaskProviderReference(
            kind=TaskProviderReferenceKind.HOSTED_URL,
            provider="anthropic",
            reference="https://private.example.test/raw",
            owner_scope="tenant-a",
            expires_at=expires_at,
            mime_type="application/pdf",
            size_bucket="1mb",
            identity_hmac="hmac-value",
            durable=False,
            metadata={"filename": "private.pdf"},
        )

        summary = reference.summary()
        execution_metadata = reference.execution_metadata()

        self.assertEqual(summary["kind"], "hosted_url")
        self.assertEqual(summary["provider"], "anthropic")
        self.assertEqual(summary["owner_scope"], "tenant-a")
        self.assertEqual(summary["metadata"], {"privacy": "<redacted>"})
        self.assertNotIn("reference", summary)
        self.assertEqual(
            execution_metadata["reference"],
            "https://private.example.test/raw",
        )
        self.assertFalse(reference.durable_for_queue)
        self.assertFalse(reference.is_expired(datetime.now(UTC)))
        self.assertNotIn("private.example", str(summary))
        self.assertNotIn("private.pdf", str(summary))

    def test_input_file_summary_redacts_private_logical_path(self) -> None:
        private_file = TaskInputFile(
            logical_path="uploads/private.pdf",
            media_type="application/pdf",
            size_bytes=100,
            metadata={"filename": "private.pdf"},
        )
        artifact_file = TaskInputFile(logical_path="artifact:artifact-1")

        private_summary = private_file.summary()

        self.assertEqual(
            private_summary["logical_path"], {"privacy": "<redacted>"}
        )
        self.assertEqual(
            private_summary["metadata"], {"privacy": "<redacted>"}
        )
        self.assertEqual(
            artifact_file.summary()["logical_path"], "artifact:artifact-1"
        )
        self.assertNotIn("uploads/private.pdf", str(private_summary))
        self.assertNotIn("private.pdf", str(private_summary))

    def test_file_input_accepts_descriptor_and_mapping_shapes(self) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.file(
                conversions=("markdown",),
                mime_types=("text/plain",),
            ),
            limits=TaskLimitsPolicy(file_bytes=20),
        )
        descriptor = TaskFileDescriptor.local_path(
            "uploads/private.txt",
            role="source",
            mime_type="text/plain",
            size_bytes=12,
            sha256="b" * 64,
            conversions=(
                TaskFileConversionRequest(
                    name="markdown",
                    options={"mode": "strict"},
                ),
            ),
            metadata={"caller": "private"},
        )
        mapping_descriptor = {
            "source_kind": "local_path",
            "reference": "uploads/second.txt",
            "mime_type": "text/plain",
            "conversions": ["markdown"],
            "metadata": {"page_count": 1},
        }

        self.assertEqual(validate_task_input(definition, descriptor), ())
        self.assertEqual(
            validate_task_input(definition, mapping_descriptor), ()
        )

    def test_file_input_accepts_provider_reference_descriptor(self) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.file(
                mime_types=("application/pdf",),
            ),
        )
        descriptor = TaskFileDescriptor.provider_reference_descriptor(
            "file-openai",
            kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
            provider="openai",
            owner_scope="tenant-a",
            mime_type="application/pdf",
            size_bucket="small",
            identity_hmac="hmac-value",
        )

        issues = validate_task_input(definition, descriptor)

        self.assertEqual(issues, ())

        mapping_issues = validate_task_input(
            definition,
            {
                "source_kind": "provider_reference",
                "reference": "file-openai",
                "mime_type": "application/pdf",
                "provider_reference": {
                    "kind": "provider_file_id",
                    "provider": "openai",
                    "reference": "file-openai",
                },
            },
        )

        self.assertEqual(mapping_issues, ())

    def test_provider_reference_descriptor_rejects_conflicts_safely(
        self,
    ) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.file(
                conversions=("text",),
                mime_types=("application/pdf",),
            ),
        )
        descriptor = {
            "source_kind": "provider_reference",
            "reference": "raw-secret-handle",
            "mime_type": "application/pdf",
            "conversions": ["text"],
            "provider_reference": {
                "kind": "not-valid",
                "provider": "bad provider",
                "reference": "other-secret-handle",
                "durable": "yes",
                "expires_at": 1,
                "mime_type": "not mime",
                "metadata": {"bad": object()},
            },
        }

        issues = validate_task_input(definition, descriptor)

        self.assertEqual(
            [issue.path for issue in issues],
            [
                "input.conversions",
                "input.provider_reference.kind",
                "input.provider_reference.provider",
                "input.provider_reference.reference",
                "input.provider_reference.durable",
                "input.provider_reference.expires_at",
                "input.provider_reference.mime_type",
                "input.provider_reference.metadata",
            ],
        )
        rendered = " ".join(
            value for issue in issues for value in issue.as_dict().values()
        )
        self.assertNotIn("raw-secret-handle", rendered)
        self.assertNotIn("other-secret-handle", rendered)

    def test_provider_reference_validation_covers_boundary_shapes(
        self,
    ) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.file(),
        )
        foreign_reference = TaskProviderReference(
            kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
            provider="openai",
            reference="different-handle",
        )

        wrong_source_issues = validate_task_input(
            definition,
            {
                "source_kind": "local_path",
                "reference": "file-openai",
                "provider_reference": {
                    "kind": "provider_file_id",
                    "provider": "openai",
                    "reference": "file-openai",
                },
            },
        )
        missing_reference_issues = validate_task_input(
            definition,
            {
                "source_kind": "provider_reference",
                "reference": "file-openai",
            },
        )
        object_reference_issues = validate_task_input(
            definition,
            {
                "source_kind": "provider_reference",
                "reference": "file-openai",
                "provider_reference": foreign_reference,
            },
        )
        metadata_issues = validate_task_input(
            definition,
            {
                "source_kind": "provider_reference",
                "reference": "file-openai",
                "provider_reference": {
                    "kind": TaskProviderReferenceKind.PROVIDER_FILE_ID,
                    "provider": "openai",
                    "reference": "file-openai",
                    "owner_scope": "",
                    "size_bucket": 1,
                    "identity_hmac": object(),
                },
            },
        )
        non_string_kind_issues = validate_task_input(
            definition,
            {
                "source_kind": "provider_reference",
                "reference": "file-openai",
                "provider_reference": {
                    "kind": 1,
                    "provider": "openai",
                    "reference": "file-openai",
                },
            },
        )

        self.assertEqual(
            [issue.path for issue in wrong_source_issues],
            ["input.provider_reference"],
        )
        self.assertEqual(
            [issue.path for issue in missing_reference_issues],
            ["input.provider_reference"],
        )
        self.assertEqual(
            [issue.path for issue in object_reference_issues],
            ["input.provider_reference.reference"],
        )
        self.assertEqual(
            [issue.path for issue in metadata_issues],
            [
                "input.provider_reference.owner_scope",
                "input.provider_reference.size_bucket",
                "input.provider_reference.identity_hmac",
            ],
        )
        self.assertEqual(
            [issue.path for issue in non_string_kind_issues],
            ["input.provider_reference.kind"],
        )
        rendered = " ".join(
            value
            for issue in (
                wrong_source_issues
                + missing_reference_issues
                + object_reference_issues
                + metadata_issues
                + non_string_kind_issues
            )
            for value in issue.as_dict().values()
        )
        self.assertNotIn("different-handle", rendered)

    def test_file_array_input_validates_count_and_descriptor_rules(
        self,
    ) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.file_array(
                conversions=("markdown",),
                mime_types=("text/plain",),
            ),
            limits=TaskLimitsPolicy(file_count=1, file_bytes=10),
        )
        value = [
            {
                "source_kind": "remote_url",
                "reference": "https://private.example.test/secret.pdf",
                "role": "bad role",
                "mime_type": "application/pdf",
                "size_bytes": 99,
                "sha256": "SECRET",
                "conversions": [
                    {
                        "name": "ocr",
                        "options": {"mode": object()},
                    },
                ],
                "metadata": {"filename": object()},
            },
            {
                "source_kind": "unknown",
                "reference": "",
                "mime_type": "text/html",
                "conversions": "markdown",
            },
        ]

        issues = validate_task_input(definition, value)

        self.assertEqual(
            {issue.code for issue in issues},
            {
                "feature.remote_url_file_inputs_disabled",
                "input.invalid_file",
            },
        )
        self.assertEqual(
            [issue.path for issue in issues],
            [
                "input",
                "input[0].source_kind",
                "input[0].role",
                "input[0].mime_type",
                "input[0].size_bytes",
                "input[0].sha256",
                "input[0].conversions[0]",
                "input[0].conversions[0].options",
                "input[0].metadata",
                "input[1].source_kind",
                "input[1].reference",
                "input[1].mime_type",
                "input[1].conversions",
            ],
        )
        rendered = " ".join(
            value for issue in issues for value in issue.as_dict().values()
        )
        self.assertNotIn("private.example", rendered)
        self.assertNotIn("secret.pdf", rendered)
        self.assertNotIn("SECRET", rendered)
        self.assertNotIn("filename", rendered)

    def test_file_contract_validation_rejects_invalid_filters(self) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.file(
                conversions=("bad conversion",),
                mime_types=("plain-text",),
            )
        )

        issues = validate_task_definition(definition)

        self.assertEqual(
            [issue.path for issue in issues],
            ["input.mime_types[0]", "input.file_conversions[0]"],
        )
        self.assertEqual(
            {issue.code for issue in issues}, {"input.invalid_file"}
        )

    def test_file_array_requires_array_and_non_empty_required_value(
        self,
    ) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.file_array()
        )

        scalar_issues = validate_task_input(definition, "uploads/private.txt")
        empty_issues = validate_task_input(definition, [])

        self.assertEqual(
            [issue.code for issue in scalar_issues],
            ["input.invalid_type"],
        )
        self.assertEqual(
            [issue.code for issue in empty_issues],
            ["input.invalid_file"],
        )

    def test_file_descriptor_validation_rejects_malformed_shapes(
        self,
    ) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.file(),
        )
        malformed_descriptor = {
            "source_kind": 1,
            "reference": "uploads/private.txt",
            "mime_type": "not mime",
            "size_bytes": False,
            "conversions": [{"name": 1}],
        }

        object_issues = validate_task_input(definition, object())
        descriptor_issues = validate_task_input(
            definition, malformed_descriptor
        )

        self.assertEqual(
            [issue.path for issue in object_issues],
            ["input"],
        )
        self.assertEqual(
            [issue.path for issue in descriptor_issues],
            [
                "input.source_kind",
                "input.mime_type",
                "input.size_bytes",
                "input.conversions[0]",
            ],
        )
        self.assertEqual(
            {issue.code for issue in object_issues + descriptor_issues},
            {"input.invalid_file"},
        )

    def test_remote_url_descriptor_is_rejected_by_default(self) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.file(),
        )
        policy = TaskRemoteUrlPolicy.disabled()

        issues = validate_task_input(
            definition,
            TaskFileDescriptor.remote_url(
                "https://private.example.test/secret.txt"
            ),
            remote_url_policy=policy,
        )

        self.assertEqual(
            [issue.code for issue in issues],
            ["feature.remote_url_file_inputs_disabled"],
        )
        self.assertEqual(
            [issue.path for issue in issues],
            ["input.source_kind"],
        )
        rendered = " ".join(
            value for issue in issues for value in issue.as_dict().values()
        )
        self.assertNotIn("private.example", rendered)
        self.assertNotIn("secret.txt", rendered)

    def test_enabled_remote_url_policy_rejects_malformed_urls(self) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.file(),
        )
        policy = TaskRemoteUrlPolicy(enabled=True, max_bytes=1024)

        invalid_port_issues = validate_task_input(
            definition,
            TaskFileDescriptor.remote_url("https://example.test:bad/input"),
            remote_url_policy=policy,
        )
        missing_host_issues = validate_task_input(
            definition,
            TaskFileDescriptor.remote_url("https:///input"),
            remote_url_policy=policy,
        )

        self.assertEqual(
            [issue.path for issue in invalid_port_issues],
            ["input.reference"],
        )
        self.assertEqual(
            [issue.path for issue in missing_host_issues],
            ["input.reference"],
        )
        rendered = " ".join(
            value
            for issue in invalid_port_issues + missing_host_issues
            for value in issue.as_dict().values()
        )
        self.assertNotIn("example.test", rendered)
        self.assertNotIn("/input", rendered)

    def test_enabled_remote_url_policy_validates_ssrf_controls(self) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.file(),
        )

        issues = validate_task_input(
            definition,
            TaskFileDescriptor.remote_url(
                "http://user:pass@127.0.0.1/private.txt"
            ),
            remote_url_policy=TaskRemoteUrlPolicy(
                enabled=True,
                allow_redirects=True,
            ),
        )

        self.assertEqual(
            [issue.path for issue in issues],
            [
                "input.reference",
                "input.reference",
                "input.reference",
                "input.redirects",
                "input.size_bytes",
            ],
        )
        self.assertEqual(
            [issue.code for issue in issues],
            ["input.invalid_file"] * 5,
        )
        rendered = " ".join(
            value for issue in issues for value in issue.as_dict().values()
        )
        self.assertNotIn("127.0.0.1", rendered)
        self.assertNotIn("user:pass", rendered)

    def test_remote_url_policy_validates_redirect_limits(self) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.file(),
        )

        missing_limit_issues = validate_task_input(
            definition,
            TaskFileDescriptor.remote_url("https://example.test/input.txt"),
            remote_url_policy=TaskRemoteUrlPolicy(
                enabled=True,
                allow_redirects=True,
                max_bytes=1024,
            ),
        )
        bounded_issues = validate_task_input(
            definition,
            TaskFileDescriptor.remote_url("https://example.test/input.txt"),
            remote_url_policy=TaskRemoteUrlPolicy(
                enabled=True,
                allow_redirects=True,
                max_redirects=2,
                max_bytes=1024,
            ),
        )

        self.assertEqual(
            [issue.path for issue in missing_limit_issues],
            ["input.redirects"],
        )
        self.assertEqual(bounded_issues, ())

    def test_enabled_remote_url_policy_rejects_local_hostnames(self) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.file(),
        )

        for url in (
            "https://localhost/private.txt",
            "https://worker.localhost/private.txt",
            "https://printer.local/private.txt",
            "https://10.0.0.1/private.txt",
            "https://172.16.0.1/private.txt",
            "https://192.168.1.2/private.txt",
            "https://169.254.1.1/private.txt",
            "https://[::1]/private.txt",
            "https://[fc00::1]/private.txt",
            "https://[fe80::1]/private.txt",
        ):
            with self.subTest(url=url):
                issues = validate_task_input(
                    definition,
                    TaskFileDescriptor.remote_url(url),
                    remote_url_policy=TaskRemoteUrlPolicy(
                        enabled=True,
                        max_bytes=1024,
                    ),
                )

                self.assertEqual(
                    [issue.path for issue in issues],
                    ["input.reference"],
                )
                rendered = " ".join(
                    value
                    for issue in issues
                    for value in issue.as_dict().values()
                )
                self.assertNotIn("localhost", rendered)
                self.assertNotIn("printer.local", rendered)
                self.assertNotIn("10.0.0.1", rendered)
                self.assertNotIn("fc00", rendered)

    def test_enabled_remote_url_policy_accepts_bounded_public_url(
        self,
    ) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.file(),
        )

        issues = validate_task_input(
            definition,
            TaskFileDescriptor.remote_url(
                "https://example.test/input.txt",
                mime_type="text/plain",
            ),
            remote_url_policy=TaskRemoteUrlPolicy(
                enabled=True,
                max_bytes=1024,
            ),
        )

        self.assertEqual(issues, ())

    def test_remote_url_policy_rejects_invalid_timeouts(self) -> None:
        for timeout in (0, -1, float("inf"), True):
            with self.subTest(timeout=timeout):
                with self.assertRaises(AssertionError):
                    TaskRemoteUrlPolicy(
                        enabled=True,
                        max_bytes=1024,
                        timeout_seconds=timeout,
                    )

    def test_descriptor_construction_rejects_unsafe_metadata(self) -> None:
        with self.assertRaises(AssertionError):
            TaskFileConversionRequest(name="ocr", options={"bad": object()})
        with self.assertRaises(AssertionError):
            TaskFileDescriptor.local_path(
                "uploads/private.txt",
                metadata={"bad": object()},
            )

    def _definition(
        self,
        *,
        input_contract: TaskInputContract,
        limits: TaskLimitsPolicy | None = None,
    ) -> TaskDefinition:
        return TaskDefinition(
            task=TaskMetadata(name="file_task", version="1"),
            input=input_contract,
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.agent("agents/file_task.toml"),
            limits=limits or TaskLimitsPolicy(),
        )


if __name__ == "__main__":
    main()
