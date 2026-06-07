from argparse import ArgumentParser, _SubParsersAction
from pathlib import Path
from unittest import TestCase, main

from avalan.cli.__main__ import CLI
from avalan.model import (
    FileDeliveryMode,
    LocalFileDeliveryProfile,
    resolve_file_delivery_profile,
)

DOC_ROOT = Path(__file__).parents[2] / "docs"
CLI_DOC = DOC_ROOT / "CLI.md"
FILE_DELIVERY_DOC = DOC_ROOT / "task_file_delivery.md"
FLOW_COMPATIBILITY_DOC = DOC_ROOT / "FLOW_COMPATIBILITY.md"
OPERATIONS_DOC = DOC_ROOT / "TASK_OPERATIONS.md"

TASK_INPUT_FLAGS = (
    "--input",
    "--input-json",
    "--input-FIELD",
    "--file",
    "--file-descriptor",
    "--provider-file-id",
    "--hosted-url",
    "--object-store-uri",
    "--file-mime",
    "--file-role",
    "--file-size",
    "--file-sha256",
    "--file-conversion",
)
TASK_RUN_FLAGS = (
    "--json",
    "--output",
    "--pdf",
)
DELIVERY_MODE_PHRASES = {
    FileDeliveryMode.PROVIDER_FILE_ID: "provider file id",
    FileDeliveryMode.HOSTED_URL: "hosted url",
    FileDeliveryMode.OBJECT_STORE_URI: "object-store uri",
    FileDeliveryMode.INLINE_BYTES: "inline bytes",
    FileDeliveryMode.INLINE_IMAGE: "inline image",
    FileDeliveryMode.INLINE_TEXT: "inline text",
    FileDeliveryMode.CONVERTED_ARTIFACT: "converted artifact",
    FileDeliveryMode.RETRIEVAL_CONTEXT: "retrieval context",
    FileDeliveryMode.MAP_REDUCE_CONTEXT: "map-reduce context",
    FileDeliveryMode.REJECT: "none",
}


class TaskDocsTest(TestCase):
    def test_cli_docs_cover_task_subcommands_and_file_flags(self) -> None:
        docs = CLI_DOC.read_text(encoding="utf-8")
        parser = CLI._create_parser("cpu", "/cache", "/locale", "en_US")
        task_parser = _find_parser(parser, " task")

        subcommands = _subcommands(task_parser)
        self.assertEqual(
            subcommands,
            {
                "artifacts",
                "enqueue",
                "events",
                "inspect",
                "output",
                "pgsql",
                "retention-sweep",
                "run",
                "usage",
                "validate",
                "worker",
            },
        )
        for command in sorted(subcommands):
            with self.subTest(command=command):
                self.assertIn(f"avalan task {command}", docs)

        run_help = _find_parser(parser, " task run").format_help()
        for flag in TASK_INPUT_FLAGS:
            with self.subTest(flag=flag):
                self.assertIn(flag, docs)
                if flag != "--input-FIELD":
                    self.assertIn(flag, run_help)
        for flag in TASK_RUN_FLAGS:
            with self.subTest(flag=flag):
                self.assertIn(flag, docs)
                self.assertIn(flag, run_help)

    def test_file_delivery_matrix_matches_profile_vocabulary(self) -> None:
        docs = FILE_DELIVERY_DOC.read_text(encoding="utf-8")
        rows = _matrix_rows(docs)
        expected = {
            "OpenAI": resolve_file_delivery_profile(
                "ai://env:KEY@openai/gpt-4o"
            ),
            "Anthropic": resolve_file_delivery_profile(
                "ai://env:KEY@anthropic/claude"
            ),
            "Google/Gemini": resolve_file_delivery_profile(
                "ai://env:KEY@google/gemini"
            ),
            "Bedrock": resolve_file_delivery_profile(
                "ai://env:KEY@bedrock/us.anthropic.claude"
            ),
            "Local text": resolve_file_delivery_profile("ai://local/model"),
            "Local multimodal": resolve_file_delivery_profile(
                "ai://local/model",
                local_profile=LocalFileDeliveryProfile.MULTIMODAL,
            ),
            "Unknown": resolve_file_delivery_profile("ai://env:KEY@vendor/m"),
        }

        self.assertEqual(set(rows), set(expected))
        for label, profile in expected.items():
            row = rows[label].lower()
            with self.subTest(profile=label):
                for mode in profile.delivery_modes:
                    self.assertIn(DELIVERY_MODE_PHRASES[mode], row)
                for scheme in profile.object_store_uri_schemes:
                    self.assertIn(f"`{scheme}`", row)
                self.assertIn(profile.name, row.replace(" ", "_"))

    def test_runbooks_cover_file_failure_modes_safely(self) -> None:
        docs = OPERATIONS_DOC.read_text(encoding="utf-8")
        required_headings = (
            "Missing Store",
            "Missing Artifact Root",
            "Unsafe Paths",
            "Remote URL Disabled Or Rejected",
            "Missing Converter",
            "Over-Limit Files",
            "Provider Credentials",
            "Local Dependency Failures",
            "Queue Payload Privacy Failures",
        )

        for heading in required_headings:
            with self.subTest(heading=heading):
                self.assertIn(f"### {heading}", docs)
        self.assertNotIn("file_abc123", docs)
        self.assertNotIn("postgresql://user:password@", docs)

    def test_flow_compatibility_docs_scope_native_flow_support(self) -> None:
        docs = FLOW_COMPATIBILITY_DOC.read_text(encoding="utf-8")
        index = (DOC_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("[Native flow compatibility]", index)
        self.assertIn("The required hosted extraction path remains", docs)
        self.assertIn("`avalan task run`", docs)
        self.assertIn("registered built-in nodes", docs)
        self.assertIn("field-addressed `--input-name`", docs)
        self.assertIn(
            "File and file-array task inputs are passed through", docs
        )
        self.assertIn("Agent nodes reuse the task agent runner", docs)
        self.assertIn("Dynamic Python callable imports.", docs)
        self.assertIn("support is available for compatible definitions", docs)
        self.assertIn("## Target Boundary", docs)
        self.assertIn(
            "`Flow.parse_mermaid(...)` is a legacy topology importer only",
            docs,
        )
        self.assertIn(
            "declarative routing and must not import callables",
            docs,
        )
        self.assertIn(
            "explicitly named and documented as compatibility",
            docs,
        )

    def test_flow_compatibility_matrix_covers_poc_fields(self) -> None:
        docs = FLOW_COMPATIBILITY_DOC.read_text(encoding="utf-8")
        rows = _flow_matrix_rows(docs)
        expected_fields = {
            "`flow.entrypoint`",
            "`flow.output_node`",
            "`flow.input.type`",
            "`flow.input.delivery`",
            "`flow.input.memory`",
            "`flow.output.schema_ref`",
            "`nodes.<name>.type`",
            "`nodes.<name>.ref`",
            "`nodes.<name>.input`",
            "`nodes.<name>.output`",
            "`nodes.<name>.user_prompt_ref`",
            "`nodes.<name>.response_format_ref`",
            "`cli.runner`",
            "`cli.example_pdf`",
        }

        self.assertTrue(expected_fields.issubset(rows))
        rejected_fields = {
            "`flow.input.delivery`",
            "`flow.input.memory`",
            "`nodes.<name>.user_prompt_ref`",
            "`nodes.<name>.response_format_ref`",
            "`cli.runner`",
            "`cli.example_pdf`",
        }
        for field in rejected_fields:
            with self.subTest(field=field):
                self.assertIn("Rejected", rows[field])


def _find_parser(parser: ArgumentParser, prog_suffix: str) -> ArgumentParser:
    stack = [parser]
    while stack:
        candidate = stack.pop()
        if candidate.prog.endswith(prog_suffix):
            return candidate
        for action in candidate._actions:
            if isinstance(action, _SubParsersAction):
                stack.extend(action.choices.values())
    raise AssertionError(f"Parser ending with {prog_suffix!r} was not found.")


def _subcommands(parser: ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, _SubParsersAction):
            return set(action.choices)
    raise AssertionError(f"Parser {parser.prog!r} has no subcommands.")


def _matrix_rows(docs: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in docs.splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 7 or cells[0] in {"Profile", "---"}:
            continue
        rows[cells[0]] = line
    return rows


def _flow_matrix_rows(docs: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in docs.splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 4 or cells[0] in {"TOML field", "---", "Gap"}:
            continue
        rows[cells[0]] = line
    return rows


if __name__ == "__main__":
    main()
