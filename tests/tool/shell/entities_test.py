from dataclasses import FrozenInstanceError
from unittest import TestCase, main

from avalan.tool.shell.entities import (
    ExecutionSpec,
    GeneratedFile,
    GeneratedOutputPlan,
    PathOperand,
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellOutputKind,
)


class ShellEntitiesTest(TestCase):
    def test_status_values_are_locked(self) -> None:
        self.assertEqual(
            {status.value for status in ShellExecutionStatus},
            {
                "completed",
                "no_matches",
                "nonzero_exit",
                "policy_denied",
                "command_unavailable",
                "spawn_failed",
                "timeout",
                "cancelled",
                "binary_skipped",
                "too_large",
                "tool_error",
            },
        )

    def test_error_code_values_cover_statuses_and_policy_reasons(
        self,
    ) -> None:
        self.assertLessEqual(
            {status.value for status in ShellExecutionStatus},
            {code.value for code in ShellExecutionErrorCode},
        )
        self.assertEqual(
            {code.value for code in ShellExecutionErrorCode}
            - {status.value for status in ShellExecutionStatus},
            {
                "denied_command",
                "denied_path",
                "traversal",
                "hidden_path",
                "sensitive_path",
                "symlink",
                "special_file",
                "binary_content",
                "too_many_arguments",
                "argument_too_large",
                "command_too_large",
                "glob_too_large",
                "invalid_option",
                "invalid_cwd",
                "write_denied",
                "shell_denied",
                "stdin_denied",
                "executable_unavailable",
                "unsafe_filter",
                "unsupported_jq_feature",
                "invalid_page_range",
                "pdf_page_cap_exceeded",
                "raster_dpi_cap_exceeded",
                "generated_output_cap_exceeded",
                "unsupported_ocr_language",
                "invalid_ocr_mode",
                "unsupported_media_signature",
            },
        )

    def test_output_kind_values_are_locked(self) -> None:
        self.assertEqual(
            {kind.value for kind in ShellOutputKind},
            {"text", "json", "generated_files"},
        )

    def test_path_operand_accepts_expected_fields(self) -> None:
        operand = PathOperand(
            name="input",
            path="src/main.py",
            kind="text_file",
            access="read",
        )

        self.assertEqual(operand.name, "input")
        self.assertTrue(operand.required)

    def test_path_operand_rejects_invalid_fields(self) -> None:
        for kwargs in (
            {"name": "", "path": "file", "kind": "file", "access": "read"},
            {"name": "input", "path": "", "kind": "file", "access": "read"},
            {
                "name": "input",
                "path": "file",
                "kind": "invalid",
                "access": "read",
            },
            {
                "name": "input",
                "path": "file",
                "kind": "file",
                "access": "invalid",
            },
            {
                "name": "input",
                "path": "file",
                "kind": "file",
                "access": "read",
                "required": 1,
            },
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(AssertionError):
                    PathOperand(**kwargs)  # type: ignore[arg-type]

    def test_shell_command_request_copies_mutable_inputs(self) -> None:
        options = {"line_numbers": True}
        metadata = {"source": "test"}
        operand = PathOperand(
            name="input",
            path="file.txt",
            kind="text_file",
            access="read",
        )
        request = ShellCommandRequest(
            tool_name="shell.rg",
            command="rg",
            options=options,
            paths=(operand,),
            cwd=".",
            metadata=metadata,
        )

        options["line_numbers"] = False
        metadata["source"] = "changed"

        self.assertEqual(request.options, {"line_numbers": True})
        self.assertEqual(request.paths, (operand,))
        self.assertEqual(request.metadata, {"source": "test"})

    def test_shell_command_request_rejects_invalid_fields(self) -> None:
        valid = {
            "tool_name": "shell.rg",
            "command": "rg",
            "options": {},
            "paths": (),
            "cwd": None,
        }
        for kwargs in (
            {"tool_name": ""},
            {"command": ""},
            {
                "options": [],
            },
            {
                "paths": [],
            },
            {
                "paths": (object(),),
            },
            {
                "cwd": "",
            },
            {
                "stdin": "input",
            },
            {
                "timeout_seconds": 0,
            },
            {
                "max_stdout_bytes": True,
            },
            {
                "metadata": [],
            },
        ):
            with self.subTest(kwargs=kwargs):
                request_kwargs = dict(valid)
                request_kwargs.update(kwargs)
                with self.assertRaises(AssertionError):
                    ShellCommandRequest(
                        **request_kwargs,
                    )  # type: ignore[arg-type]

    def test_execution_spec_copies_mutable_inputs(self) -> None:
        env = {"LC_ALL": "C"}
        metadata = {"clamped": True}
        spec = ExecutionSpec(
            backend="local",
            tool_name="shell.rg",
            command="rg",
            executable="/usr/bin/rg",
            argv=("rg", "needle"),
            display_argv=("rg", "needle"),
            cwd="/workspace",
            display_cwd=".",
            env=env,
            stdin=None,
            stdout_media_type="text/plain",
            output_kind=ShellOutputKind.TEXT,
            resource_class="standard",
            output_plan=None,
            timeout_seconds=1.0,
            max_stdout_bytes=10,
            max_stderr_bytes=11,
            metadata=metadata,
        )

        env["LC_ALL"] = "changed"
        metadata["clamped"] = False

        self.assertEqual(spec.env, {"LC_ALL": "C"})
        self.assertEqual(spec.metadata, {"clamped": True})

    def test_execution_spec_rejects_invalid_fields(self) -> None:
        valid = {
            "backend": "local",
            "tool_name": "shell.rg",
            "command": "rg",
            "executable": "/usr/bin/rg",
            "argv": ("rg",),
            "display_argv": ("rg",),
            "cwd": "/workspace",
            "display_cwd": ".",
            "env": {},
            "stdin": None,
            "stdout_media_type": "text/plain",
            "output_kind": ShellOutputKind.TEXT,
            "resource_class": "standard",
            "output_plan": None,
            "timeout_seconds": 1.0,
            "max_stdout_bytes": 10,
            "max_stderr_bytes": 10,
        }
        invalid_values = {
            "backend": "remote",
            "tool_name": "",
            "command": "",
            "executable": "",
            "argv": ["rg"],
            "display_argv": ["rg"],
            "cwd": "",
            "display_cwd": "",
            "env": {"": "value"},
            "stdin": "input",
            "stdout_media_type": "",
            "output_kind": "text",
            "resource_class": "unknown",
            "output_plan": object(),
            "timeout_seconds": 0,
            "max_stdout_bytes": -1,
            "max_stderr_bytes": True,
            "metadata": [],
        }
        for field_name, value in invalid_values.items():
            with self.subTest(field_name=field_name):
                kwargs = dict(valid)
                kwargs[field_name] = value
                with self.assertRaises(AssertionError):
                    ExecutionSpec(**kwargs)  # type: ignore[arg-type]

    def test_generated_output_plan_copies_mutable_inputs(self) -> None:
        suffixes = [".png"]
        media_types = {".png": "image/png"}
        plan = GeneratedOutputPlan(
            prefix="page",
            display_prefix="page",
            allowed_suffixes=tuple(suffixes),
            suffix_media_types=media_types,
            max_files=1,
            max_file_bytes=2,
            max_total_bytes=3,
            max_inline_bytes=4,
        )

        suffixes.append(".jpg")
        media_types[".png"] = "application/octet-stream"

        self.assertEqual(plan.allowed_suffixes, (".png",))
        self.assertEqual(plan.suffix_media_types, {".png": "image/png"})

    def test_generated_output_plan_is_frozen(self) -> None:
        plan = GeneratedOutputPlan(
            prefix="page",
            display_prefix="page",
            allowed_suffixes=(".png",),
            suffix_media_types={".png": "image/png"},
            max_files=1,
            max_file_bytes=2,
            max_total_bytes=3,
            max_inline_bytes=4,
        )

        with self.assertRaises(FrozenInstanceError):
            plan.prefix = "other"

    def test_generated_file_defaults_and_metadata_copy(self) -> None:
        metadata = {"page": 1}
        generated_file = GeneratedFile(
            display_path="page-1.png",
            media_type="image/png",
            suffix=".png",
            bytes=10,
            metadata=metadata,
        )

        metadata["page"] = 2

        self.assertIsNone(generated_file.sha256)
        self.assertIsNone(generated_file.page)
        self.assertIsNone(generated_file.width)
        self.assertIsNone(generated_file.height)
        self.assertIsNone(generated_file.content_base64)
        self.assertFalse(generated_file.truncated)
        self.assertEqual(generated_file.metadata, {"page": 1})

    def test_generated_file_accepts_optional_metadata_fields(self) -> None:
        generated_file = GeneratedFile(
            display_path="page-1.png",
            media_type="image/png",
            suffix=".png",
            bytes=10,
            sha256="a" * 64,
            page=1,
            width=2,
            height=3,
            content_base64="YWJj",
            truncated=True,
        )

        self.assertEqual(generated_file.sha256, "a" * 64)
        self.assertEqual(generated_file.content_base64, "YWJj")
        self.assertTrue(generated_file.truncated)

    def test_generated_file_is_frozen(self) -> None:
        generated_file = GeneratedFile(
            display_path="page-1.png",
            media_type="image/png",
            suffix=".png",
            bytes=10,
        )

        with self.assertRaises(FrozenInstanceError):
            generated_file.bytes = 20


if __name__ == "__main__":
    main()
