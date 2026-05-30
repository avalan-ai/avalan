from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import cast
from unittest import TestCase, main

from avalan.task import (
    TaskDefinition,
    TaskExecutionTarget,
    TaskFileConversionRequest,
    TaskFileDescriptor,
    TaskFileSourceKind,
    TaskInputContract,
    TaskLimitsPolicy,
    TaskMetadata,
    TaskOutputContract,
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
        )

        self.assertEqual(
            [descriptor.source_kind for descriptor in descriptors],
            [
                TaskFileSourceKind.LOCAL_PATH,
                TaskFileSourceKind.REMOTE_URL,
                TaskFileSourceKind.ARTIFACT,
                TaskFileSourceKind.INLINE_BYTES,
            ],
        )

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

    def test_enabled_remote_url_policy_rejects_local_hostnames(self) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.file(),
        )

        for url in (
            "https://localhost/private.txt",
            "https://worker.localhost/private.txt",
            "https://printer.local/private.txt",
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
