from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

from rich.console import Console

from avalan.cli.__main__ import CLI, _consume_task_input_field_args
from avalan.cli.commands import task as task_cmds
from avalan.task import (
    PrivacyAction,
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskPrivacyPolicy,
    validate_task_input,
)

TASK_HMAC_ENV = {
    "AVALAN_TASK_HMAC_KEY_ID": "cli-test-v1",
    "AVALAN_TASK_HMAC_KEY_B64": "dGFzay1obWFjLXRlc3Qta2V5",
}


class CliTaskInputParserTestCase(TestCase):
    def test_parser_accepts_task_input_options(self) -> None:
        parser = CLI._create_parser("cpu", "/cache", "/locale", "en_US")

        args = parser.parse_args(
            [
                "task",
                "enqueue",
                "tasks/document.task.toml",
                "--queue",
                "priority-documents",
                "--input-json",
                "@input.json",
                "--input-question",
                "What are the risks?",
                "--input-priority=2",
                "--file",
                "documents=report-a.pdf",
                "--file",
                "documents=report-b.pdf",
            ]
        )

        self.assertEqual(args.command, "task")
        self.assertEqual(args.task_command, "enqueue")
        self.assertEqual(args.definition, "tasks/document.task.toml")
        self.assertEqual(args.queue, "priority-documents")
        self.assertEqual(args.task_input_json, "@input.json")
        self.assertEqual(
            args.task_input_fields,
            ("question=What are the risks?", "priority=2"),
        )
        self.assertEqual(
            args.task_files,
            ["documents=report-a.pdf", "documents=report-b.pdf"],
        )

    def test_parser_accepts_explicit_file_descriptor_options(self) -> None:
        parser = CLI._create_parser("cpu", "/cache", "/locale", "en_US")

        args = parser.parse_args(
            [
                "task",
                "run",
                "tasks/document.task.toml",
                "--provider-file-id",
                "document=openai:file-private",
                "--hosted-url",
                "references=openai:https://example.test/private.pdf",
                "--object-store-uri",
                "references=google:gs://bucket/private.pdf",
                "--file-mime",
                "document=application/pdf",
                "--file-role",
                "document=source",
                "--file-size",
                "document=2048",
                "--file-sha256",
                "document=" + ("a" * 64),
                "--file-conversion",
                'document=text:{"encoding":"utf-8"}',
                "--json",
                "--output",
                "result.json",
                "--pdf",
                "input.pdf",
            ]
        )

        self.assertEqual(
            args.task_provider_file_ids,
            ["document=openai:file-private"],
        )
        self.assertEqual(
            args.task_hosted_urls,
            ["references=openai:https://example.test/private.pdf"],
        )
        self.assertEqual(
            args.task_object_store_uris,
            ["references=google:gs://bucket/private.pdf"],
        )
        self.assertEqual(
            args.task_file_mime_types,
            ["document=application/pdf"],
        )
        self.assertEqual(args.task_file_roles, ["document=source"])
        self.assertEqual(args.task_file_sizes, ["document=2048"])
        self.assertEqual(args.task_file_sha256, ["document=" + ("a" * 64)])
        self.assertEqual(
            args.task_file_conversions,
            ['document=text:{"encoding":"utf-8"}'],
        )
        self.assertTrue(args.task_run_json)
        self.assertEqual(args.task_output_path, "result.json")
        self.assertEqual(args.task_pdf, "input.pdf")

    def test_parser_keeps_existing_model_input_file_behavior(self) -> None:
        parser = CLI._create_parser("cpu", "/cache", "/locale", "en_US")

        args = parser.parse_args(
            [
                "model",
                "run",
                "model-id",
                "--input-file",
                "doc-1.pdf",
                "--input-file",
                "doc-2.pdf",
            ]
        )

        self.assertEqual(args.input_file, ["doc-1.pdf", "doc-2.pdf"])

    def test_parser_rejects_dynamic_input_outside_task_inputs(self) -> None:
        parser = CLI._create_parser("cpu", "/cache", "/locale", "en_US")

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "model",
                    "run",
                    "model-id",
                    "--input-question",
                    "raw",
                ]
            )

    def test_dynamic_input_rejects_invalid_shapes(self) -> None:
        cases = (
            (
                Namespace(command="model", task_command=None),
                ["--input-question", "raw"],
            ),
            (
                Namespace(command="task", task_command="inspect"),
                ["--input-question", "raw"],
            ),
            (
                Namespace(command="task", task_command="run"),
                ["--not-input", "raw"],
            ),
            (
                Namespace(command="task", task_command="run"),
                ["--input-question"],
            ),
            (
                Namespace(command="task", task_command="run"),
                ["--input-question", "--file"],
            ),
            (
                Namespace(command="task", task_command="run"),
                ["--input-1bad", "raw"],
            ),
        )

        for namespace, extras in cases:
            with self.subTest(extras=extras):
                self.assertFalse(
                    _consume_task_input_field_args(namespace, extras)
                )


class CliTaskInputTestCase(TestCase):
    def test_absent_input_returns_unprovided_value(self) -> None:
        parsed = task_cmds.task_cli_input(
            Namespace(
                task_input=None,
                task_input_json=None,
                task_input_fields=(),
                task_files=(),
            ),
            self._definition(TaskInputContract.string()),
        )

        self.assertFalse(parsed.provided)
        self.assertIsNone(parsed.value)

    def test_plain_input_matches_scalar_contract(self) -> None:
        parsed = task_cmds.task_cli_input(
            Namespace(
                task_input="Leo Messi",
                task_input_json=None,
                task_input_fields=(),
                task_files=(),
            ),
            self._definition(TaskInputContract.string()),
        )

        self.assertTrue(parsed.provided)
        self.assertEqual(parsed.value, "Leo Messi")
        self.assertEqual(
            validate_task_input(
                self._definition(TaskInputContract.string()),
                parsed.value,
            ),
            (),
        )

    def test_json_file_input_matches_object_contract(self) -> None:
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.json"
            input_path.write_text(
                '{"question":"What changed?","priority":2}',
                encoding="utf-8",
            )

            parsed = task_cmds.task_cli_input(
                Namespace(
                    task_input=None,
                    task_input_json=f"@{input_path}",
                    task_input_fields=(),
                    task_files=(),
                ),
                self._definition(self._object_contract()),
            )

        self.assertEqual(
            parsed.value,
            {"question": "What changed?", "priority": 2},
        )
        self.assertEqual(
            validate_task_input(
                self._definition(self._object_contract()),
                parsed.value,
            ),
            (),
        )

    def test_json_object_merges_field_and_file_inputs(self) -> None:
        definition = self._definition(self._object_contract())

        parsed = task_cmds.task_cli_input(
            Namespace(
                task_input=None,
                task_input_json='{"nested":{"count":1},"documents":[]}',
                task_input_fields=("nested.label=review",),
                task_files=("documents=report.pdf",),
            ),
            definition,
        )

        self.assertEqual(
            parsed.value,
            {
                "nested": {"count": 1, "label": "review"},
                "documents": [
                    {"source_kind": "local_path", "reference": "report.pdf"}
                ],
            },
        )
        self.assertEqual(validate_task_input(definition, parsed.value), ())

    def test_field_and_file_input_builds_valid_object(self) -> None:
        definition = self._definition(self._object_contract())

        parsed = task_cmds.task_cli_input(
            Namespace(
                task_input=None,
                task_input_json=None,
                task_input_fields=("question=What changed?", "priority=2"),
                task_files=(
                    "nested.document=brief.pdf",
                    "documents=report-a.pdf",
                    "documents=report-b.pdf",
                ),
            ),
            definition,
        )

        self.assertEqual(
            parsed.value,
            {
                "question": "What changed?",
                "priority": 2,
                "nested": {
                    "document": {
                        "source_kind": "local_path",
                        "reference": "brief.pdf",
                    },
                },
                "documents": [
                    {
                        "source_kind": "local_path",
                        "reference": "report-a.pdf",
                    },
                    {
                        "source_kind": "local_path",
                        "reference": "report-b.pdf",
                    },
                ],
            },
        )
        self.assertEqual(validate_task_input(definition, parsed.value), ())

    def test_pdf_and_file_input_build_file_array_object_fields(self) -> None:
        definition = self._definition(
            TaskInputContract.object(
                schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pdf": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["source_kind", "reference"],
                                "properties": {
                                    "source_kind": {"type": "string"},
                                    "reference": {"type": "string"},
                                    "mime_type": {"type": "string"},
                                },
                            },
                            "x-avalan-input-type": "file[]",
                            "x-avalan-mime-types": ["application/pdf"],
                        },
                        "image": {"type": "boolean"},
                    },
                }
            )
        )

        pdf_input = task_cmds.task_cli_input(
            Namespace(
                task_input=None,
                task_input_json=None,
                task_pdf="sample.pdf",
                task_input_fields=("image=true",),
                task_files=(),
            ),
            definition,
        )
        file_input = task_cmds.task_cli_input(
            Namespace(
                task_input=None,
                task_input_json=None,
                task_pdf=None,
                task_input_fields=("image=true",),
                task_files=("pdf=sample.pdf",),
            ),
            definition,
        )

        self.assertEqual(
            pdf_input.value,
            {
                "image": True,
                "pdf": [
                    {
                        "source_kind": "local_path",
                        "reference": "sample.pdf",
                        "mime_type": "application/pdf",
                    }
                ],
            },
        )
        self.assertEqual(
            file_input.value,
            {
                "image": True,
                "pdf": [
                    {
                        "source_kind": "local_path",
                        "reference": "sample.pdf",
                    }
                ],
            },
        )
        self.assertEqual(validate_task_input(definition, pdf_input.value), ())
        self.assertEqual(validate_task_input(definition, file_input.value), ())

    def test_single_file_input_builds_valid_descriptor(self) -> None:
        definition = self._definition(TaskInputContract.file())

        parsed = task_cmds.task_cli_input(
            Namespace(
                task_input=None,
                task_input_json=None,
                task_input_fields=(),
                task_files=("document=report.pdf",),
            ),
            definition,
        )

        self.assertEqual(
            parsed.value,
            {"source_kind": "local_path", "reference": "report.pdf"},
        )
        self.assertEqual(validate_task_input(definition, parsed.value), ())

    def test_file_array_input_builds_valid_descriptor_list(self) -> None:
        definition = self._definition(TaskInputContract.file_array())

        parsed = task_cmds.task_cli_input(
            Namespace(
                task_input=None,
                task_input_json=None,
                task_input_fields=(),
                task_files=("documents=report-a.pdf", "documents=b.pdf"),
            ),
            definition,
        )

        self.assertEqual(
            parsed.value,
            [
                {"source_kind": "local_path", "reference": "report-a.pdf"},
                {"source_kind": "local_path", "reference": "b.pdf"},
            ],
        )
        self.assertEqual(validate_task_input(definition, parsed.value), ())

    def test_file_input_applies_descriptor_hints_and_conversions(
        self,
    ) -> None:
        definition = self._definition(
            TaskInputContract.file(
                conversions=("text",),
                mime_types=("application/pdf",),
            )
        )
        digest = "a" * 64

        parsed = task_cmds.task_cli_input(
            Namespace(
                task_input=None,
                task_input_json=None,
                task_input_fields=(),
                task_files=("document=report.pdf",),
                task_file_descriptors=(),
                task_provider_file_ids=(),
                task_hosted_urls=(),
                task_object_store_uris=(),
                task_file_mime_types=("document=application/pdf",),
                task_file_roles=("document=source",),
                task_file_sizes=("document=1024",),
                task_file_sha256=(f"document={digest}",),
                task_file_conversions=('document=text:{"encoding":"utf-8"}',),
            ),
            definition,
        )

        self.assertEqual(
            parsed.value,
            {
                "source_kind": "local_path",
                "reference": "report.pdf",
                "mime_type": "application/pdf",
                "role": "source",
                "size_bytes": 1024,
                "sha256": digest,
                "conversions": [
                    {"name": "text", "options": {"encoding": "utf-8"}}
                ],
            },
        )
        self.assertEqual(validate_task_input(definition, parsed.value), ())

    def test_provider_reference_file_inputs_build_valid_descriptors(
        self,
    ) -> None:
        definition = self._definition(TaskInputContract.file_array())

        parsed = task_cmds.task_cli_input(
            Namespace(
                task_input=None,
                task_input_json=None,
                task_input_fields=(),
                task_files=(),
                task_file_descriptors=(),
                task_provider_file_ids=("document=openai:file-private",),
                task_hosted_urls=(
                    "url=openai:https://example.test/private.pdf",
                ),
                task_object_store_uris=(
                    "object=google:gs://bucket/private.pdf",
                ),
                task_file_mime_types=("document=application/pdf",),
                task_file_roles=(),
                task_file_sizes=(),
                task_file_sha256=(),
                task_file_conversions=(),
            ),
            definition,
        )

        self.assertEqual(
            parsed.value,
            [
                {
                    "source_kind": "provider_reference",
                    "reference": "file-private",
                    "provider_reference": {
                        "kind": "provider_file_id",
                        "provider": "openai",
                        "reference": "file-private",
                    },
                    "mime_type": "application/pdf",
                },
                {
                    "source_kind": "provider_reference",
                    "reference": "https://example.test/private.pdf",
                    "provider_reference": {
                        "kind": "hosted_url",
                        "provider": "openai",
                        "reference": "https://example.test/private.pdf",
                    },
                },
                {
                    "source_kind": "provider_reference",
                    "reference": "gs://bucket/private.pdf",
                    "provider_reference": {
                        "kind": "object_store_uri",
                        "provider": "google",
                        "reference": "gs://bucket/private.pdf",
                    },
                },
            ],
        )
        self.assertEqual(validate_task_input(definition, parsed.value), ())

    def test_explicit_json_descriptor_merges_with_object_input(self) -> None:
        definition = self._definition(self._object_contract())

        parsed = task_cmds.task_cli_input(
            Namespace(
                task_input=None,
                task_input_json='{"question":"What changed?"}',
                task_input_fields=(),
                task_files=(),
                task_file_descriptors=(
                    (
                        'document={"source_kind":"remote_url",'
                        '"reference":"https://example.test/file.txt",'
                        '"mime_type":"text/plain"}'
                    ),
                ),
                task_provider_file_ids=(),
                task_hosted_urls=(),
                task_object_store_uris=(),
                task_file_mime_types=(),
                task_file_roles=(),
                task_file_sizes=(),
                task_file_sha256=(),
                task_file_conversions=(),
            ),
            definition,
        )

        self.assertEqual(
            parsed.value,
            {
                "question": "What changed?",
                "document": {
                    "source_kind": "remote_url",
                    "reference": "https://example.test/file.txt",
                    "mime_type": "text/plain",
                },
            },
        )

    def test_scalar_input_coercion_variants(self) -> None:
        cases = (
            (TaskInputContract.integer(), "7", 7),
            (TaskInputContract.number(), "7.5", 7.5),
            (TaskInputContract.boolean(), "true", True),
            (
                self._object_contract(),
                '{"question":"What changed?"}',
                {"question": "What changed?"},
            ),
            (
                TaskInputContract.array(schema={"type": "array"}),
                "[1,2]",
                [1, 2],
            ),
            (
                TaskInputContract.file(),
                "report.pdf",
                {"source_kind": "local_path", "reference": "report.pdf"},
            ),
            (
                TaskInputContract.file_array(),
                "report.pdf",
                [{"source_kind": "local_path", "reference": "report.pdf"}],
            ),
        )

        for contract, raw_value, expected in cases:
            with self.subTest(contract=contract.type):
                parsed = task_cmds.task_cli_input(
                    Namespace(
                        task_input=raw_value,
                        task_input_json=None,
                        task_input_fields=(),
                        task_files=(),
                    ),
                    self._definition(contract),
                )

            self.assertEqual(parsed.value, expected)

    def test_parse_errors_are_safe(self) -> None:
        cases = (
            (
                Namespace(
                    task_input="raw",
                    task_input_json="{}",
                    task_input_fields=(),
                    task_files=(),
                ),
                self._definition(TaskInputContract.string()),
            ),
            (
                Namespace(
                    task_input="raw",
                    task_input_json=None,
                    task_input_fields=("question=value",),
                    task_files=(),
                ),
                self._definition(TaskInputContract.string()),
            ),
            (
                Namespace(
                    task_input=None,
                    task_input_json='"scalar"',
                    task_input_fields=("question=value",),
                    task_files=(),
                ),
                self._definition(self._object_contract()),
            ),
            (
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=("question=value",),
                    task_files=("document=report.pdf",),
                ),
                self._definition(TaskInputContract.file()),
            ),
            (
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=("a=one.pdf", "b=two.pdf"),
                ),
                self._definition(TaskInputContract.file()),
            ),
            (
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    task_file_descriptors=("document",),
                ),
                self._definition(TaskInputContract.file()),
            ),
            (
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    task_file_descriptors=("document=[]",),
                ),
                self._definition(TaskInputContract.file()),
            ),
            (
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    task_provider_file_ids=("document=openai",),
                ),
                self._definition(TaskInputContract.file()),
            ),
            (
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    task_file_mime_types=("document=application/pdf",),
                ),
                self._definition(TaskInputContract.file()),
            ),
            (
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=("document=report.pdf",),
                    task_file_descriptors=(),
                    task_file_conversions=("document=:{}",),
                ),
                self._definition(TaskInputContract.file()),
            ),
            (
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=("document=report.pdf",),
                    task_file_descriptors=(),
                    task_file_conversions=("document=text:[]",),
                ),
                self._definition(TaskInputContract.file()),
            ),
            (
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=("document=report.pdf",),
                    task_file_sizes=("document=abc",),
                ),
                self._definition(TaskInputContract.file()),
            ),
            (
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=("document=report.pdf",),
                    task_file_mime_types=(
                        "document=application/pdf",
                        "document=text/plain",
                    ),
                ),
                self._definition(TaskInputContract.file()),
            ),
            (
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    task_file_descriptors=(
                        (
                            'document={"source_kind":"local_path",'
                            '"reference":"report.pdf",'
                            '"conversions":"text"}'
                        ),
                    ),
                    task_file_conversions=("document=text",),
                ),
                self._definition(TaskInputContract.file()),
            ),
            (
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    task_file_descriptors=(
                        (
                            'document={"source_kind":"local_path",'
                            '"reference":"report.pdf",'
                            '"mime_type":"application/pdf"}'
                        ),
                    ),
                    task_file_mime_types=("document=text/plain",),
                ),
                self._definition(TaskInputContract.file()),
            ),
            (
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=("document=report.pdf",),
                    task_file_roles=("document",),
                ),
                self._definition(TaskInputContract.file()),
            ),
        )

        for args, definition in cases:
            with self.subTest(args=args):
                with self.assertRaises(task_cmds.TaskCliInputError) as error:
                    task_cmds.task_cli_input(args, definition)

            self.assertEqual(error.exception.code, "input.parse")
            self.assertNotIn("report.pdf", error.exception.message)

    def test_invalid_scalar_values_are_safe(self) -> None:
        cases = (
            (TaskInputContract.integer(), "true"),
            (TaskInputContract.number(), "NaN"),
            (TaskInputContract.boolean(), "1"),
            (self._object_contract(), "{"),
        )

        for contract, raw_value in cases:
            with self.subTest(contract=contract.type):
                with self.assertRaises(task_cmds.TaskCliInputError):
                    task_cmds.task_cli_input(
                        Namespace(
                            task_input=raw_value,
                            task_input_json=None,
                            task_input_fields=(),
                            task_files=(),
                        ),
                        self._definition(contract),
                    )

    def test_invalid_field_and_file_flags_are_safe(self) -> None:
        cases = (
            Namespace(
                task_input=None,
                task_input_json=None,
                task_input_fields=("badfield",),
                task_files=(),
            ),
            Namespace(
                task_input=None,
                task_input_json=None,
                task_input_fields=(),
                task_files=("document",),
            ),
        )

        for args in cases:
            with self.subTest(args=args):
                with self.assertRaises(task_cmds.TaskCliInputError):
                    task_cmds.task_cli_input(
                        args,
                        self._definition(self._object_contract()),
                    )

    def test_json_read_and_json_parse_errors_are_safe(self) -> None:
        cases = (
            ("@/tmp/private/input.json", "input.read"),
            ("@", "input.parse"),
            ("{", "input.json"),
        )

        for raw_json, code in cases:
            with self.subTest(raw_json=raw_json):
                with self.assertRaises(task_cmds.TaskCliInputError) as error:
                    task_cmds.task_cli_input(
                        Namespace(
                            task_input=None,
                            task_input_json=raw_json,
                            task_input_fields=(),
                            task_files=(),
                        ),
                        self._definition(self._object_contract()),
                    )

            self.assertEqual(error.exception.code, code)
            self.assertNotIn(
                "/tmp/private/input.json", error.exception.message
            )

    def test_field_conflicts_and_duplicates_are_rejected(self) -> None:
        cases = (
            Namespace(
                task_input=None,
                task_input_json='{"a":1}',
                task_input_fields=("a.b=2",),
                task_files=(),
            ),
            Namespace(
                task_input=None,
                task_input_json=None,
                task_input_fields=("a=1", "a=2"),
                task_files=(),
            ),
        )

        for args in cases:
            with self.subTest(args=args):
                with self.assertRaises(task_cmds.TaskCliInputError):
                    task_cmds.task_cli_input(
                        args,
                        self._definition(self._object_contract()),
                    )

    def test_validate_prints_redacted_input_summary(self) -> None:
        console = Console(record=True, width=160)
        with TemporaryDirectory() as tmpdir:
            definition = Path(tmpdir) / "question.task.toml"
            definition.write_text(
                """
                [task]
                name = "question"
                version = "1"

                [input]
                type = "object"

                [input.schema]
                type = "object"
                required = ["question"]

                [input.schema.properties.question]
                type = "string"

                [output]
                type = "text"

                [execution]
                type = "agent"
                ref = "agents/question.toml"
                """,
                encoding="utf-8",
            )

            with patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True):
                result = task_cmds.task_validate(
                    Namespace(
                        definition=str(definition),
                        task_input=None,
                        task_input_json=None,
                        task_input_fields=("question=private question",),
                        task_files=(),
                    ),
                    console,
                    object(),
                )

        output = console.export_text()
        self.assertTrue(result)
        self.assertIn("Task input is valid.", output)
        self.assertIn("<redacted>", output)
        self.assertNotIn("private question", output)

    def test_validate_reports_input_parse_and_validation_errors(self) -> None:
        with TemporaryDirectory() as tmpdir:
            definition = self._write_definition(
                tmpdir,
                name="integer",
                input_type="integer",
            )
            parse_console = Console(record=True, width=160)
            with patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True):
                parse_result = task_cmds.task_validate(
                    Namespace(
                        definition=str(definition),
                        task_input="not-an-integer",
                        task_input_json=None,
                        task_input_fields=(),
                        task_files=(),
                    ),
                    parse_console,
                    object(),
                )

            validation_console = Console(record=True, width=160)
            with patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True):
                validation_result = task_cmds.task_validate(
                    Namespace(
                        definition=str(definition),
                        task_input=None,
                        task_input_json=None,
                        task_input_fields=(),
                        task_files=("document=report.pdf",),
                    ),
                    validation_console,
                    object(),
                )

        self.assertFalse(parse_result)
        self.assertIn(
            "Task input could not be parsed.",
            parse_console.export_text(),
        )
        self.assertFalse(validation_result)
        self.assertIn(
            "Task input is invalid.", validation_console.export_text()
        )

    def test_validate_rejects_provider_reference_conversion_safely(
        self,
    ) -> None:
        console = Console(record=True, width=160)
        with TemporaryDirectory() as tmpdir:
            definition = Path(tmpdir) / "document.task.toml"
            definition.write_text(
                """
                [task]
                name = "document"
                version = "1"

                [input]
                type = "file"
                file_conversions = ["text"]

                [output]
                type = "text"

                [execution]
                type = "agent"
                ref = "agents/document.toml"
                """,
                encoding="utf-8",
            )

            with patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True):
                result = task_cmds.task_validate(
                    Namespace(
                        definition=str(definition),
                        task_input=None,
                        task_input_json=None,
                        task_input_fields=(),
                        task_files=(),
                        task_provider_file_ids=(
                            "document=openai:file-private",
                        ),
                        task_file_conversions=("document=text",),
                    ),
                    console,
                    object(),
                )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task input is invalid.", output)
        self.assertIn("input.invalid_file", output)
        self.assertNotIn("file-private", output)

    def test_run_reports_input_validation_before_store_resolution(
        self,
    ) -> None:
        console = Console(record=True, width=160)
        with TemporaryDirectory() as tmpdir:
            definition = Path(tmpdir) / "integer.task.toml"
            definition.write_text(
                """
                [task]
                name = "integer"
                version = "1"

                [input]
                type = "integer"

                [output]
                type = "text"

                [execution]
                type = "agent"
                ref = "agents/integer.toml"
                """,
                encoding="utf-8",
            )

            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input="not-an-integer",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                ),
                console,
                object(),
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task input could not be parsed.", output)
        self.assertNotIn("not-an-integer", output)

    def test_run_accepts_valid_input_before_store_resolution(self) -> None:
        console = Console(record=True, width=160)
        with TemporaryDirectory() as tmpdir:
            definition = self._write_definition(
                tmpdir,
                name="integer",
                input_type="integer",
            )

            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input="7",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=False,
                ),
                console,
                object(),
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task store is not configured.", output)
        self.assertIn("store.missing", output)

    def test_enqueue_reports_invalid_input_before_store_resolution(
        self,
    ) -> None:
        console = Console(record=True, width=160)
        with TemporaryDirectory() as tmpdir:
            definition = self._write_definition(
                tmpdir,
                name="integer",
                input_type="integer",
            )

            result = task_cmds.task_enqueue(
                Namespace(
                    definition=str(definition),
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=("document=report.pdf",),
                ),
                console,
                object(),
            )

        self.assertFalse(result)
        self.assertIn("Task input is invalid.", console.export_text())

    def test_command_input_validation_handles_load_errors(self) -> None:
        missing_console = Console(record=True, width=160)
        missing_result = task_cmds.task_run(
            Namespace(
                definition="/tmp/private/missing.task.toml",
                task_input="raw",
                task_input_json=None,
                task_input_fields=(),
                task_files=(),
            ),
            missing_console,
            object(),
        )

        with TemporaryDirectory() as tmpdir:
            malformed = Path(tmpdir) / "bad.task.toml"
            malformed.write_text('[task]\nname = "bad"\n', encoding="utf-8")
            load_console = Console(record=True, width=160)
            load_result = task_cmds.task_run(
                Namespace(
                    definition=str(malformed),
                    task_input="raw",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                ),
                load_console,
                object(),
            )

        self.assertFalse(missing_result)
        self.assertIn("file.read", missing_console.export_text())
        self.assertFalse(load_result)
        self.assertIn(
            "Task definition could not be loaded.",
            load_console.export_text(),
        )

    def test_input_validation_helper_allows_missing_definition_attribute(
        self,
    ) -> None:
        console = Console(record=True, width=160)

        result = task_cmds._validate_task_cli_input_for_command(
            Namespace(
                task_input="raw",
                task_input_json=None,
                task_input_fields=(),
                task_files=(),
            ),
            console,
        )

        self.assertTrue(result)
        self.assertEqual(console.export_text(), "")

    def test_summary_helpers_cover_files_and_redaction_fallback(self) -> None:
        console = Console(record=True, width=160)
        definition = self._definition(TaskInputContract.file_array())
        value = [
            {"source_kind": "local_path", "reference": "report-a.pdf"},
            {"source_kind": "local_path", "reference": "report-b.pdf"},
        ]

        task_cmds._print_task_cli_input_summary(console, definition, value)
        rendered = console.export_text()

        self.assertIn("file[0]", rendered)
        self.assertIn("file[1]", rendered)
        self.assertNotIn("report-a.pdf", rendered)
        self.assertEqual(
            task_cmds._format_task_cli_value(("a", "b")),
            '["a","b"]',
        )

        fallback_console = Console(record=True, width=160)
        fallback_definition = self._definition(
            TaskInputContract.string(),
            privacy=TaskPrivacyPolicy(input=PrivacyAction.HASH),
        )
        task_cmds._print_task_cli_input_summary(
            fallback_console,
            fallback_definition,
            "private input",
        )

        self.assertIn("<redacted>", fallback_console.export_text())
        self.assertNotIn("private input", fallback_console.export_text())

    def test_pdf_input_sugar_builds_pdf_descriptor(self) -> None:
        definition = self._definition(
            TaskInputContract.file(mime_types=("application/pdf",))
        )
        array_definition = self._definition(
            TaskInputContract.file_array(mime_types=("application/pdf",))
        )

        task_input = task_cmds.task_cli_input(
            Namespace(
                task_input=None,
                task_input_json=None,
                task_pdf="sample.pdf",
                task_input_fields=(),
                task_files=(),
            ),
            definition,
        )
        array_input = task_cmds.task_cli_input(
            Namespace(
                task_input=None,
                task_input_json=None,
                task_pdf="sample.pdf",
                task_input_fields=(),
                task_files=(),
            ),
            array_definition,
        )

        self.assertTrue(task_input.provided)
        self.assertEqual(
            task_input.value,
            {
                "source_kind": "local_path",
                "reference": "sample.pdf",
                "mime_type": "application/pdf",
            },
        )
        self.assertEqual(array_input.value, [task_input.value])

    def test_pdf_input_sugar_rejects_conflicts_and_non_file_contracts(
        self,
    ) -> None:
        file_definition = self._definition(TaskInputContract.file())
        file_array_definition = self._definition(
            TaskInputContract.file_array()
        )
        string_definition = self._definition(TaskInputContract.string())
        object_definition = self._definition(self._object_contract())
        cases = (
            (
                file_definition,
                Namespace(
                    task_input="raw",
                    task_input_json=None,
                    task_pdf="sample.pdf",
                    task_input_fields=(),
                    task_files=(),
                ),
                "Pass --pdf by itself",
            ),
            (
                file_definition,
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_pdf="sample.pdf",
                    task_input_fields=(),
                    task_files=("input=other.pdf",),
                ),
                "Pass --pdf by itself",
            ),
            (
                file_definition,
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_pdf="sample.pdf",
                    task_input_fields=(),
                    task_files=(),
                    task_file_mime_types=("input=text/plain",),
                ),
                "Pass --pdf by itself",
            ),
            (
                file_array_definition,
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_pdf="sample.pdf",
                    task_input_fields=("image=true",),
                    task_files=(),
                ),
                "Pass --pdf by itself",
            ),
            (
                file_array_definition,
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_pdf="sample.pdf",
                    task_input_fields=(),
                    task_files=(),
                    task_file_mime_types=("input=text/plain",),
                ),
                "Pass --pdf by itself",
            ),
            (
                object_definition,
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_pdf="sample.pdf",
                    task_input_fields=(),
                    task_files=(),
                ),
                "single top-level file input",
            ),
            (
                string_definition,
                Namespace(
                    task_input=None,
                    task_input_json=None,
                    task_pdf="sample.pdf",
                    task_input_fields=(),
                    task_files=(),
                ),
                "single top-level file input",
            ),
        )

        for definition, args, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(task_cmds.TaskCliInputError) as error:
                    task_cmds.task_cli_input(args, definition)

            self.assertIn(message, str(error.exception))

    def test_pdf_input_schema_helpers_cover_guard_paths(self) -> None:
        no_schema_definition = self._definition(TaskInputContract.object())
        duplicate_pdf_definition = self._definition(
            TaskInputContract.object(
                schema={
                    "type": "object",
                    "properties": {
                        "front": {
                            "x-avalan-input-type": "file",
                            "x-avalan-mime-types": ["application/pdf"],
                        },
                        "back": {
                            "x-avalan-input-type": "file",
                            "x-avalan-mime-types": ["application/pdf"],
                        },
                    },
                }
            )
        )
        nested_scalar_definition = self._definition(
            TaskInputContract.object(
                schema={
                    "type": "object",
                    "properties": {"pdf": "not an object"},
                }
            )
        )
        missing_field_definition = self._definition(
            TaskInputContract.object(
                schema={"type": "object", "properties": {}}
            )
        )
        bad_extension_definition = self._definition(
            TaskInputContract.object(
                schema={
                    "type": "object",
                    "properties": {"pdf": {"x-avalan-input-type": 1}},
                }
            )
        )

        self.assertIsNone(
            task_cmds._task_cli_pdf_object_field(no_schema_definition)
        )
        self.assertIsNone(
            task_cmds._task_cli_pdf_object_field(duplicate_pdf_definition)
        )
        self.assertIsNone(
            task_cmds._task_cli_schema_field_input_type(
                no_schema_definition,
                "pdf",
            )
        )
        self.assertIsNone(
            task_cmds._task_cli_schema_field_input_type(
                nested_scalar_definition,
                "pdf.child",
            )
        )
        self.assertIsNone(
            task_cmds._task_cli_schema_field_input_type(
                missing_field_definition,
                "pdf",
            )
        )
        self.assertIsNone(
            task_cmds._task_cli_schema_field_input_type(
                bad_extension_definition,
                "pdf",
            )
        )
        self.assertFalse(task_cmds._task_cli_schema_accepts_pdf("pdf"))
        self.assertTrue(
            task_cmds._task_cli_schema_accepts_pdf(
                {"x-avalan-input-type": "file"}
            )
        )
        self.assertFalse(
            task_cmds._task_cli_schema_accepts_pdf(
                {
                    "x-avalan-input-type": "file",
                    "x-avalan-mime-types": "application/pdf",
                }
            )
        )

    def test_unsupported_input_type_is_rejected(self) -> None:
        definition = self._definition(TaskInputContract.string())
        object.__setattr__(definition.input, "type", "unknown")

        with self.assertRaises(task_cmds.TaskCliInputError):
            task_cmds.task_cli_input(
                Namespace(
                    task_input="raw",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                ),
                definition,
            )

    def _definition(
        self,
        input_contract: TaskInputContract,
        *,
        privacy: TaskPrivacyPolicy | None = None,
    ) -> TaskDefinition:
        return TaskDefinition(
            task=TaskMetadata(name="task", version="1"),
            input=input_contract,
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.agent("agents/task.toml"),
            privacy=privacy or TaskPrivacyPolicy(),
        )

    def _object_contract(self) -> TaskInputContract:
        return TaskInputContract.object(schema={"type": "object"})

    def _write_definition(
        self,
        directory: str,
        *,
        name: str,
        input_type: str,
    ) -> Path:
        definition = Path(directory) / f"{name}.task.toml"
        definition.write_text(
            f"""
            [task]
            name = "{name}"
            version = "1"

            [input]
            type = "{input_type}"

            [output]
            type = "text"

            [execution]
            type = "agent"
            ref = "agents/{name}.toml"
            """,
            encoding="utf-8",
        )
        return definition


if __name__ == "__main__":
    main()
